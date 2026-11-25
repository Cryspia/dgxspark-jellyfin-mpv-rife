import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dual_machine"))
from queue_mgr import FrameVA, QueueManager, TaskState, TaskType


class WorkerRetryTest(unittest.TestCase):
    def test_speculative_neighbor_retains_output_until_owner_requests_it(self):
        manager = QueueManager()
        args = dict(sa_y_stride=16, sa_uv_stride=8, dst_y_stride=32,
                    dst_uv_stride=16, dst_y=301, dst_u=302, dst_v=303)
        manager.submit_with_phase_dst(0, 1, sa_y=11, sa_u=12, sa_v=13,
                                      sb_y=21, sb_u=22, sb_v=23, **args)
        epoch = manager.current_seek_epoch()
        backing, output = object(), object()
        source = dict(src_y=21, src_u=22, src_v=23, src_y_stride=16,
                      src_uv_stride=8, _retry_source_backing=backing)
        self.assertTrue(manager.needs_speculative_neighbor(0, epoch))
        self.assertTrue(manager.install_speculative_neighbor(0, epoch,
                                                             source=source, output=output))
        with manager._cv:
            producer = manager._dag_find_locked(TaskType.CCSR, 1)
            self.assertEqual(producer.state, TaskState.READY)
        manager.task_done(producer.id)
        self.assertNotIn(1, manager._signaled_K_set)
        manager.submit_with_phase_dst(1, 0, sa_y=101, sa_u=102, sa_v=103,
                                      sb_y=201, sb_u=202, sb_v=203, **args)
        self.assertIs(manager.speculative_output(1), output)
        self.assertIs(producer.payload["_retry_source_backing"], backing)
        self.assertEqual(producer.payload["src_y"], 21)
        self.assertTrue(manager.wait_phase_done(1, 0, timeout=0.01))
        manager.consume_speculative_output(1, output)
        self.assertIsNone(manager.speculative_output(1))

    def test_speculation_cannot_cross_seek_epoch(self):
        manager = QueueManager()
        manager.submit_frame(0, va=FrameVA(pair_k=0, sa_y=1, sa_u=2, sa_v=3,
                                           sb_y=4, sb_u=5, sb_v=6,
                                           sa_y_stride=16, sa_uv_stride=8))
        epoch = manager.current_seek_epoch()
        manager.flush_seek()
        self.assertFalse(manager.install_speculative_neighbor(0, epoch,
                                                              source={}, output=object()))

    def test_seek_retires_running_speculation_without_reusing_its_node(self):
        manager = QueueManager()
        manager.submit_frame(0, va=FrameVA(pair_k=0, sa_y=1, sa_u=2, sa_v=3,
                                           sb_y=4, sb_u=5, sb_v=6,
                                           sa_y_stride=16, sa_uv_stride=8))
        epoch = manager.current_seek_epoch()
        output = object()
        source = dict(src_y=4, src_u=5, src_v=6, src_y_stride=16,
                      src_uv_stride=8, _retry_source_backing=object())
        self.assertTrue(manager.install_speculative_neighbor(0, epoch,
                                                             source=source, output=output))
        with manager._cv:
            old = manager._dag_find_locked(TaskType.CCSR, 1)
            old.state = TaskState.RUNNING
            old.payload["_dispatch_epoch"] = epoch
        manager.flush_seek()
        manager.submit_frame(1, va=FrameVA(pair_k=1, sa_y=10, sa_u=11, sa_v=12,
                                           sb_y=13, sb_u=14, sb_v=15,
                                           sa_y_stride=16, sa_uv_stride=8))
        with manager._cv:
            replacement = manager._dag_find_locked(TaskType.CCSR, 1)
            self.assertIsNot(replacement, old)
        manager.task_done(old.id)
        with manager._cv:
            self.assertNotIn("_speculative_output", old.payload)
            self.assertNotIn(1, manager._signaled_K_set)

    def test_failed_stale_speculation_releases_private_buffers(self):
        manager = QueueManager()
        manager.submit_frame(0, va=FrameVA(pair_k=0, sa_y=1, sa_u=2, sa_v=3,
                                           sb_y=4, sb_u=5, sb_v=6,
                                           sa_y_stride=16, sa_uv_stride=8))
        epoch = manager.current_seek_epoch()
        self.assertTrue(manager.install_speculative_neighbor(
            0, epoch, source=dict(src_y=4, src_u=5, src_v=6,
                                  src_y_stride=16, src_uv_stride=8,
                                  _retry_source_backing=object()), output=object()))
        with manager._cv:
            old = manager._dag_find_locked(TaskType.CCSR, 1)
            old.state = TaskState.RUNNING
            old.payload["_dispatch_epoch"] = epoch
            slot = manager._cc_alloc_slot_locked(frame_idx=1)
            old.payload["dst_cc_slot"] = slot
        manager.flush_seek()
        manager.task_done(old.id, ok=False, error="worker lost")
        with manager._cv:
            self.assertNotIn("_speculative_output", old.payload)
            self.assertNotIn("_retry_source_backing", old.payload)
            self.assertNotIn(slot, manager._cc)

    def test_owner_can_arrive_while_speculative_producer_is_running(self):
        manager = QueueManager()
        args = dict(sa_y_stride=16, sa_uv_stride=8, dst_y_stride=32,
                    dst_uv_stride=16, dst_y=301, dst_u=302, dst_v=303)
        manager.submit_with_phase_dst(0, 1, sa_y=11, sa_u=12, sa_v=13,
                                      sb_y=21, sb_u=22, sb_v=23, **args)
        output = object()
        source = dict(src_y=21, src_u=22, src_v=23, src_y_stride=16,
                      src_uv_stride=8, _retry_source_backing=object())
        self.assertTrue(manager.install_speculative_neighbor(
            0, manager.current_seek_epoch(), source=source, output=output))
        with manager._cv:
            producer = manager._dag_find_locked(TaskType.CCSR, 1)
            producer.state = TaskState.RUNNING
        manager.submit_with_phase_dst(1, 0, sa_y=101, sa_u=102, sa_v=103,
                                      sb_y=201, sb_u=202, sb_v=203, **args)
        self.assertIs(manager.speculative_output(1), output)
        self.assertEqual(producer.payload["src_y"], 21)
        manager.task_done(producer.id)
        self.assertTrue(manager.wait_phase_done(1, 0, timeout=0.01))
        manager.consume_speculative_output(1, output)
        self.assertIsNone(manager.speculative_output(1))

    def test_seek_between_submission_and_wait_abandons_old_epoch(self):
        manager = QueueManager()
        epoch = manager.current_seek_epoch()
        args = dict(sa_y=1, sa_u=2, sa_v=3, sb_y=4, sb_u=5, sb_v=6,
                    sa_y_stride=16, sa_uv_stride=8, dst_y=7, dst_u=8, dst_v=9,
                    dst_y_stride=32, dst_uv_stride=16, expected_epoch=epoch)
        self.assertTrue(manager.submit_with_phase_dst(0, 0, **args))
        manager.flush_seek()
        self.assertEqual(manager.wait_phase_done(0, 0, timeout=0.01,
                                                 expected_epoch=epoch),
                         manager.WAIT_ABANDONED)
        self.assertFalse(manager.submit_with_phase_dst(1, 0, **args))
        with manager._cv:
            self.assertFalse(manager._dag)

    def test_interpolation_sources_follow_frame_order_not_task_order(self):
        manager = QueueManager()
        with manager._cv:
            later = manager._new_node(task_type=TaskType.CCSR, frame_idx=11)
            earlier = manager._new_node(task_type=TaskType.CCSR, frame_idx=10)
            later.payload["dst_cc_slot"] = 22
            earlier.payload["dst_cc_slot"] = 11
            interp = manager._new_node(task_type=TaskType.INTERP, frame_idx=10,
                                       parents=[earlier.id, later.id])
            self.assertTrue(manager._post_pop_locked(interp))
            self.assertEqual(interp.payload["src_cc_slot_a"], 11)
            self.assertEqual(interp.payload["src_cc_slot_b"], 22)

    def test_neighbor_placeholder_waits_for_owner_source(self):
        manager = QueueManager()
        args = dict(sa_y_stride=16, sa_uv_stride=8, dst_y_stride=32,
                    dst_uv_stride=16, dst_y=301, dst_u=302, dst_v=303)
        manager.submit_with_phase_dst(0, 1, sa_y=11, sa_u=12, sa_v=13,
                                      sb_y=999, sb_u=998, sb_v=997, **args)
        manager.set_phase_dst(1, 0, dst_y=301, dst_u=302, dst_v=303,
                              dst_y_stride=32, dst_uv_stride=16)
        with manager._cv:
            producer = manager._dag_find_locked(TaskType.CCSR, 1)
            self.assertFalse(producer.payload.get("src_y"))
            self.assertEqual(producer.state, TaskState.PENDING)
        manager.submit_with_phase_dst(1, 0, sa_y=101, sa_u=102, sa_v=103,
                                      sb_y=201, sb_u=202, sb_v=203, **args)
        self.assertEqual(producer.payload["src_y"], 101)
        self.assertEqual(producer.state, TaskState.READY)

    def test_neighbor_callback_cannot_replace_owner_source(self):
        manager = QueueManager()
        args = dict(sa_y_stride=16, sa_uv_stride=8, dst_y_stride=32,
                    dst_uv_stride=16, dst_y=301, dst_u=302, dst_v=303)
        manager.submit_with_phase_dst(1, 0, sa_y=101, sa_u=102, sa_v=103,
                                      sb_y=201, sb_u=202, sb_v=203, **args)
        manager.submit_with_phase_dst(0, 1, sa_y=11, sa_u=12, sa_v=13,
                                      sb_y=999, sb_u=998, sb_v=997, **args)
        with manager._cv:
            producer = manager._dag_find_locked(TaskType.CCSR, 1)
            self.assertEqual((producer.payload["src_y"], producer.payload["src_u"],
                              producer.payload["src_v"]), (101, 102, 103))

    def test_submit_frame_keeps_owner_source_when_neighbor_arrives_late(self):
        manager = QueueManager()
        manager.submit_frame(1, va=FrameVA(pair_k=1, sa_y=101, sa_u=102, sa_v=103,
                                           sb_y=201, sb_u=202, sb_v=203,
                                           sa_y_stride=16, sa_uv_stride=8))
        manager.submit_frame(0, va=FrameVA(pair_k=0, sa_y=11, sa_u=12, sa_v=13,
                                           sb_y=999, sb_u=998, sb_v=997,
                                           sa_y_stride=16, sa_uv_stride=8))
        with manager._cv:
            producer = manager._dag_find_locked(TaskType.CCSR, 1)
            self.assertEqual(producer.payload["src_y"], 101)

    def test_source_slot_survives_late_neighbor_submission(self):
        manager = QueueManager()
        manager.submit_frame(0, va=FrameVA(pair_k=0, sa_y=1, sa_u=2, sa_v=3,
                                           sb_y=4, sb_u=5, sb_v=6,
                                           sa_y_stride=16, sa_uv_stride=8))
        with manager._cv:
            producer = manager._dag_find_locked(TaskType.CCSR, 1)
            slot = manager._cc_alloc_slot_locked(frame_idx=1)
            producer.payload["dst_cc_slot"] = slot
        manager.task_stage_done(producer.id, 0)
        with manager._cv:
            self.assertEqual(manager._cc[slot].rgb_padded_refs, 2)
            self.assertEqual(manager._cc[slot].rife_features_refs, 2)
            manager._cc_drop_slot_field_locked(slot, "rgb_padded")
            manager._cc_drop_slot_field_locked(slot, "rife_features")
            producer.state = TaskState.DONE
            manager._dag_find_locked(TaskType.INTERP, 0).state = TaskState.DONE
            manager._gc_locked(2)
            self.assertIn(producer.id, manager._dag)
            self.assertIn(slot, manager._cc)
        manager.submit_frame(1, va=FrameVA(pair_k=1, sa_y=4, sa_u=5, sa_v=6,
                                           sb_y=7, sb_u=8, sb_v=9,
                                           sa_y_stride=16, sa_uv_stride=8))
        with manager._cv:
            late = manager._dag_find_locked(TaskType.INTERP, 1)
            self.assertIn(producer.id, late.original_parents)
            self.assertEqual(manager._cc[slot].rgb_padded_refs, 1)
            manager._cc_drop_slot_field_locked(slot, "rgb_padded")
            manager._cc_drop_slot_field_locked(slot, "rife_features")
            self.assertNotIn(slot, manager._cc)
            late.state = TaskState.DONE
            manager._gc_locked(3)
            self.assertNotIn(producer.id, manager._dag)

    def test_old_epoch_producer_does_not_reserve_future_consumers(self):
        manager = QueueManager()
        with manager._cv:
            producer = manager._new_node(task_type=TaskType.CCSR, frame_idx=4)
            slot = manager._cc_alloc_slot_locked(frame_idx=4)
            producer.payload.update(dst_cc_slot=slot, _dispatch_epoch=0)
            manager._seek_epoch = 1
            manager._initialise_refcounts_locked(producer)
            self.assertNotIn(slot, manager._cc)

    def test_seek_releases_previously_published_source_after_producer_finishes(self):
        manager = QueueManager()
        manager.submit_frame(0, va=FrameVA(pair_k=0, sa_y=1, sa_u=2, sa_v=3,
                                           sb_y=4, sb_u=5, sb_v=6,
                                           sa_y_stride=16, sa_uv_stride=8))
        with manager._cv:
            producer = manager._dag_find_locked(TaskType.CCSR, 1)
            slot = manager._cc_alloc_slot_locked(frame_idx=1)
            producer.payload.update(dst_cc_slot=slot, _dispatch_epoch=0)
            producer.state = TaskState.RUNNING
        manager.task_stage_done(producer.id, 0)
        manager.flush_seek()
        manager.task_done(producer.id)
        with manager._cv:
            self.assertNotIn(slot, manager._cc)

    def test_seek_keeps_source_until_running_consumer_finishes(self):
        manager = QueueManager()
        manager.submit_frame(0, va=FrameVA(pair_k=0, sa_y=1, sa_u=2, sa_v=3,
                                           sb_y=4, sb_u=5, sb_v=6,
                                           sa_y_stride=16, sa_uv_stride=8))
        with manager._cv:
            producer = manager._dag_find_locked(TaskType.CCSR, 1)
            consumer = manager._dag_find_locked(TaskType.INTERP, 0)
            slot = manager._cc_alloc_slot_locked(frame_idx=1)
            producer.payload.update(dst_cc_slot=slot, _dispatch_epoch=0)
            producer.state = TaskState.RUNNING
            consumer.state = TaskState.RUNNING
            consumer.payload.update(src_cc_slot_a=-1, src_cc_slot_b=slot)
        manager.task_stage_done(producer.id, 0)
        manager.flush_seek()
        manager.task_done(producer.id)
        with manager._cv:
            self.assertEqual(manager._cc[slot].rgb_padded_refs, 1)
            self.assertEqual(manager._cc[slot].rife_features_refs, 1)
        manager.task_done(consumer.id)
        with manager._cv:
            self.assertNotIn(slot, manager._cc)

    def test_new_child_accepts_already_published_parent_stages(self):
        manager = QueueManager()
        with manager._cv:
            first = manager._new_node(task_type=TaskType.CCSR, frame_idx=4)
            second = manager._new_node(task_type=TaskType.CCSR, frame_idx=5)
            first.payload["stages_fired"] = {0}
            child = manager._new_node(
                task_type=TaskType.INTERP, frame_idx=4,
                parents=[first.id, second.id],
                parents_at_stage={first.id: 0, second.id: 0})
            self.assertEqual(child.parents, {second.id})
            self.assertEqual(child.deps_remaining, 1)
        manager.task_stage_done(second.id, 0)
        self.assertEqual(child.deps_remaining, 0)
        self.assertFalse(child.parents)

    def test_retained_window_registers_both_cache_consumers(self):
        manager = QueueManager()
        with manager._lock:
            for frame in (11, 12):
                manager.submit_frame(frame, va=FrameVA(
                    pair_k=frame, sa_y=101, sa_u=102, sa_v=103,
                    sb_y=201, sb_u=202, sb_v=203,
                    sa_y_stride=16, sa_uv_stride=8))
            node = manager._dag_find_locked(TaskType.CCSR, 12)
            slot = manager._cc_alloc_slot_locked(frame_idx=12)
            node.payload["dst_cc_slot"] = slot
            manager._initialise_refcounts_locked(node)
            self.assertEqual(manager._cc[slot].rgb_padded_refs, 2)
            self.assertEqual(manager._cc[slot].rife_features_refs, 2)

    def test_midstream_session_bootstraps_its_first_source(self):
        manager = QueueManager()
        manager.submit_with_phase_dst(
            30, 0, sa_y=101, sa_u=102, sa_v=103,
            sb_y=201, sb_u=202, sb_v=203,
            sa_y_stride=16, sa_uv_stride=8,
            dst_y=301, dst_u=302, dst_v=303,
            dst_y_stride=32, dst_uv_stride=16)
        node = manager.pop_for_host(blocking=False)
        self.assertEqual((node.type, node.frame_idx, node.payload["src_y"]),
                         (TaskType.CCSR, 30, 101))

    def test_retry_preserves_published_stages_and_source(self):
        manager = QueueManager()
        with manager._cv:
            node = manager._new_node(task_type=TaskType.CCSR, frame_idx=4,
                                     payload={"src_y": 101})
            manager._frame_va[4] = FrameVA(pair_k=4, dst_a_y=301)
            manager._make_ready_locked(node)
        claimed = manager.pop_for_guest(blocking=False)
        self.assertIs(claimed, node)
        manager.task_stage_done(node.id, 0)
        backup = object()
        self.assertTrue(manager.retry_task(node.id, {"_retry_source_backing": backup}))
        self.assertEqual(node.state, TaskState.READY)
        self.assertEqual(node.payload["retry_skipped_stages"], frozenset({0}))
        self.assertIs(node.payload["_retry_source_backing"], backup)
        self.assertIs(manager.pop_for_host(blocking=False), node)
        manager.task_stage_done(node.id, 0)
        self.assertEqual(node.payload["stages_fired"], {0})

    def test_gc_keeps_destination_for_unfinished_source_frame(self):
        manager = QueueManager()
        with manager._cv:
            source = manager._new_node(task_type=TaskType.CCSR, frame_idx=4,
                                       payload={"src_y": 101})
            output = manager._new_node(task_type=TaskType.SR_INTERP, frame_idx=4)
            source.state = TaskState.RUNNING
            source.payload["_dispatch_epoch"] = manager._seek_epoch
            output.state = TaskState.DONE
            manager._frame_va[4] = FrameVA(pair_k=4, dst_a_y=301)
            manager._gc_locked(5)
            self.assertNotIn(output.id, manager._dag)
            self.assertIsNotNone(manager._frame_va.get(4))
            self.assertTrue(manager.retry_task(source.id))
            self.assertEqual(source.state, TaskState.READY)

    def test_retry_waits_for_destination_registration(self):
        manager = QueueManager()
        with manager._cv:
            node = manager._new_node(task_type=TaskType.CCSR, frame_idx=4,
                                     payload={"src_y": 101})
            manager._make_ready_locked(node)
        self.assertIs(manager.pop_for_guest(blocking=False), node)
        self.assertTrue(manager.retry_task(node.id))
        self.assertEqual(node.state, TaskState.PENDING)
        self.assertIsNone(manager.pop_for_host(blocking=False))
        manager.set_phase_dst(4, 0, dst_y=301, dst_u=302, dst_v=303,
                              dst_y_stride=32, dst_uv_stride=16)
        self.assertIs(manager.pop_for_host(blocking=False), node)

    def test_retry_rejects_pre_seek_task(self):
        manager = QueueManager()
        with manager._cv:
            node = manager._new_node(task_type=TaskType.CCSR, frame_idx=4)
            manager._make_ready_locked(node)
        self.assertIs(manager.pop_for_guest(blocking=False), node)
        manager.flush_seek()
        self.assertFalse(manager.retry_task(node.id))


if __name__ == "__main__":
    unittest.main()
