# Hot-Aware HiRadixCache 设计与实现思路

## 1. 背景与问题

当前 HiRadix + HiCache 的行为中，前缀复用和增量写入都可能长期驻留在 GPU：

1. 被前缀匹配命中的 KV，或从 Host 加载回 CUDA 的 KV，会参与 LRU（按 timestamp）管理。
2. 请求 decode 过程中新增的增量 KV，在请求结束后也会进入前缀树并保留 CUDA 副本。

这会带来一个偏置：

1. 新增增量 KV 的 timestamp 通常更“新”，在 LRU 中不容易被淘汰。
2. 历史上被频繁匹配的热前缀，反而可能更早被淘汰。

目标是把“热度”定义收敛为：

1. 热 KV：被匹配命中，或从 Host 加载到 CUDA 后被使用的 KV。
2. 冷增量 KV：请求新增的 decode 增量，在完成异步备份且请求结束后，尽快从 CUDA 释放，只保留 Host 副本。

## 2. 现有流程梳理

关键路径如下：

1. 前缀匹配与插入：python/minisgl/kvcache/hiradix_cache.py
2. Cache 管理与请求缓存：python/minisgl/scheduler/cache.py
3. HiCache 异步 load/write 与 ACK 回收：python/minisgl/hicache/controller.py
4. 请求完成时资源回收：python/minisgl/scheduler/scheduler.py

当前与问题直接相关的行为：

1. CacheManager.cache_req(finished=True) 在请求结束时会将请求 prefix 插入 HiRadix。
2. HiCacheController.prepare_write 会给 on_cuda_only 节点补齐 host_value 并异步写回 Host。
3. 写回完成后在 refresh() 中仅执行 unlock，不会主动丢弃该段 CUDA 副本。
4. 因此这段“已备份增量”仍参与 CUDA LRU，且 timestamp 较新。

## 3. 设计目标

### 3.1 功能目标

1. 热点感知：热前缀（匹配或被加载）继续由 timestamp + LRU 管理。
2. 增量冷却：decode 结束请求的新增 KV，在 Host 写回完成后主动从 CUDA 释放。
3. 行为安全：释放前必须满足“已写回完成 + 无锁引用”。

### 3.2 非目标

1. 不改变 Radix 树匹配语义。
2. 不改变已有 page table 分配模型。
3. 不引入新的跨进程通信协议。

## 4. 核心策略

### 4.1 KV 分类

按节点生命周期分为两类：

1. Hot-Managed KV（热管理）
   - 触发条件：被匹配命中；或由 Host 加载至 CUDA 后被实际参与计算。
   - 策略：保留 CUDA 副本，继续参与 LRU 淘汰。

2. Delta-Evict KV（增量可释放）
   - 触发条件：请求 decode 期间新增的 CUDA-only 节点。
   - 策略：异步写回 Host；在写回 ACK 且请求结束后，主动将对应节点切换为 host_only。

### 4.2 释放时机

必须同时满足：

1. write ACK 已完成（跨 TP rank 共识完成）。
2. 对应 handle 已解锁，目标节点 ref_count == 0。

满足后执行：

1. 回收索引池：收集将要降级节点对应的 cuda indices，并归还给 CacheManager 的 free slots（按 page_size 对齐），使槽位可再次分配。
2. 节点降级：将节点从 cuda+host 切换为 host_only（node.cuda_value = None），保留 node.host_value 作为回载来源。
3. 统计修正：按节点当前 ref_count 修正 size_info（evictable_size/protected_size），确保完整性检查与后续淘汰逻辑一致。

### 4.3 与现有 LRU 的关系

该策略不是改 LRU 排序，而是改变候选集合：

1. 热前缀继续用 timestamp/LRU。
2. 已备份完成的增量段不再占用 CUDA，也不再进入 CUDA LRU 候选。

这可以避免“新鲜增量挤压热前缀”的现象。

## 5. 建议代码改造点

## 5.1 HiRadixPrefixCache 扩展

文件：python/minisgl/kvcache/hiradix_cache.py

新增接口建议：

1. drop_cuda_suffix(handle, drop_len) -> torch.Tensor
   - 从 handle.node 向上按路径释放连续 suffix 的 CUDA 副本，并返回可回收的 cuda indices。
   - 仅允许释放已具备 host_value 的节点。
   - 返回值用于归还 CacheManager 的 free slots。

2. 可选辅助接口：collect_cuda_only_suffix(handle) -> List[node]
   - 在 set_host 或 prepare_write 阶段记录“本次写回对应节点集合”。

实现要点：

1. 释放节点前校验 node._host_value is not None。
2. 释放后校正 size_info：
   - ref_count == 0: evictable_size -= node.length
   - ref_count > 0: protected_size -= node.length
3. 不删除树节点结构，仅将 CUDA 副本去除，保持 host-only 可被后续匹配。
4. drop_cuda_suffix 不直接操作 free slots；由调用方（CacheManager/HiCacheController 协同）统一回收索引池。

## 5.2 HiCacheController 写回 ACK 增强

文件：python/minisgl/hicache/controller.py

建议扩展 Transaction/Ack 元数据，记录“是否在 ACK 后释放 CUDA 增量”：

1. prepare_write 增加参数：drop_cuda_after_write: bool
2. 对 finished 请求提交的写事务，设置 drop_cuda_after_write = True
3. refresh() 处理 ack_write_queue 时：
   - 先 unlock(handle)
   - 再按记录信息调用 hiradix_cache.drop_cuda_suffix(...)

可选数据携带方案：

1. 在 Ack 中记录每个 handle 的 drop_len。
2. 或直接记录节点引用列表（更直接，但要谨慎生命周期和可见性）。

## 5.3 CacheManager 调用侧改造

文件：python/minisgl/scheduler/cache.py

在 cache_req(req, finished=...) 中：

1. finished=False（prefill 后继续 decode）
   - 保持现状：prepare_write(new_handle, drop_cuda_after_write=False)

2. finished=True（decode 完成或 abort 回收）
   - 调用 prepare_write(new_handle, drop_cuda_after_write=True)

这样可确保“仅已完成请求的增量段”会在写回后释放 CUDA。

## 5.4 Scheduler 行为保持

文件：python/minisgl/scheduler/scheduler.py

主流程无需大改，依赖现有：

1. _free_req_resources(req) -> cache_req(req, finished=True)
2. 周期性 _schedule_next_batch() 里 refresh_hicache()

因此写回 ACK 后释放可以自然发生在 refresh() 阶段。

## 6. 关键边界条件

1. 并发安全
   - 必须在写回完成后释放，避免读写竞态。
   - 必须在 unlock 后尝试释放，避免破坏正在使用的 handle。

2. 引用计数
   - 若尝试释放时 ref_count > 0，说明被其他请求命中了热前缀，此时应
   不释放，仅释放可释放部分。

3. Host 容量不足
   - 现有 _try_allocate_host 可能返回 None。
   - 若写回未成功，不可释放 CUDA 增量，保持当前安全行为。

4. page_size 对齐
   - drop_len 与 writable_len 均需保持 page 对齐，延续当前对齐约束。

## 7. 伪代码草案

### 7.1 CacheManager

```python
cached_len, new_handle = prefix_cache.insert_prefix(...)
if enable_hicache:
    hicache_controller.prepare_write(
        new_handle,
        drop_cuda_after_write=finished,
    )
unlock(old_handle)
...
```

### 7.2 HiCacheController.refresh

```python
for ack in finished_write_acks:
    for handle, drop_len, do_drop in ack.items:
        hiradix_cache.lock_handle(handle, unlock=True)
        if do_drop:
         released = hiradix_cache.drop_cuda_suffix(handle, drop_len)
         cache_manager.free_indices(released)
```

### 7.3 HiRadixPrefixCache.drop_cuda_suffix

```python
node = handle.node
left = drop_len
released = []
while not node.is_root() and left > 0:
    if node.cuda_value is None:
        break
    assert node.host_value is not None
   released.append(node.cuda_value)
    if node.ref_count == 0:
        evictable_size -= node.length
    else:
        protected_size -= node.length
    node.cuda_value = None
    left -= node.length
    node = node.parent
return cat(released)
```

## 8. 验证计划

建议至少覆盖以下测试：

1. 热前缀保留优先级
   - 构造“高复用前缀 + 大量短请求增量”，验证热前缀命中率提升。

2. 写回后释放
   - finished 请求写回 ACK 后，目标节点应从 cuda+host 变为 host_only。
   - size_info 中 CUDA 侧容量统计应下降。

3. 不破坏正确性
   - 后续新请求匹配到 host_only 节点后可正确触发 load 并继续生成。

4. Host 不足回退
   - Host 容量不足时不发生误释放，系统行为与当前一致。

## 9. 分阶段落地建议

1. Phase 1: 最小可用版本
   - 仅增加 drop_cuda_after_write + drop_cuda_suffix 能力。
   - 先按 finished=True 请求执行释放。

2. Phase 2: 可观测性
   - 增加日志与计数指标：
     - released_cuda_tokens
     - write_ack_pending
     - host_only_nodes
> 注意代码简洁美观，可通过一个 option 参数（命令行参数）控制 Hot-Aware 的开关，便于后续对比测试

<!-- 3. Phase 3: 策略调优
   - 支持阈值化释放（例如仅释放超过 N pages 的增量）。
   - 评估是否需要对热点访问附加权重，而不仅是 timestamp。 -->


## 10. 预期收益

1. 减少 decode 增量在 GPU 的“长尾占用”。
2. 提升热前缀在 CUDA 的驻留概率与复用收益。
3. 在不改变主调度流程的前提下，增强 HiRadix 的热点感知能力。
