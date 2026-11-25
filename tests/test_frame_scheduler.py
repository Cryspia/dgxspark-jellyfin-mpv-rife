import unittest
import random

from dual_machine.frame_scheduler import FrameScheduler, Kind, OutputReorder


class FrameSchedulerTest(unittest.TestCase):
    def test_out_of_order_gap_keeps_two_worker_ownership_feasible(self):
        s = FrameScheduler(['w1', 'w2'], prefetch=True,
                           fused_ccsr=True, frame_slots=3)
        s.submit_interval(15)
        self.assertEqual(s.claim('w1').frame, 15)
        s.submit_interval(13)  # Reveals missing frame 14 between 13 and 15.
        self.assertIsNone(s.claim('w2'))
        self.assertEqual(s.claim_ahead('w1').frame, 13)
        self.assertEqual(s.claim('w2').frame, 14)

    def test_two_workers_keep_an_owner_for_unassigned_neighbor(self):
        s = FrameScheduler(['w1', 'w2'], prefetch=True,
                           fused_ccsr=True, frame_slots=3)
        s.submit_interval(16)
        s.submit_interval(18)
        self.assertEqual(s.claim('w1').frame, 16)
        # w1 cannot take 17 and must not skip it to take 18; doing so can
        # prevent either worker from owning an earlier missing neighbor.
        self.assertIsNone(s.claim_ahead('w1'))
        self.assertEqual(s.claim('w2').frame, 17)
        self.assertEqual(s.claim_ahead('w1').frame, 18)

    def test_ahead_source_is_direct_and_nonadjacent(self):
        s = FrameScheduler(['w1', 'w2'], prefetch=True, fused_ccsr=True)
        s.submit_interval(0)
        s.submit_interval(1)
        current = s.claim('w1')
        s.claim('w2')
        ahead = s.claim_ahead('w1')
        self.assertEqual((current.frame, ahead.frame), (0, 2))
        self.assertEqual(ahead.kind, Kind.CCSR)
        self.assertFalse(ahead.prefetched)
        # Completion of the worker's early feature stage is valid even
        # if the previous task's completion notice is still in flight.
        self.assertTrue(s.report_prepared(ahead))
        self.assertTrue(s._frames[2].prepared)
        self.assertNotEqual(s.cache_slots('w1', 0), s.cache_slots('w1', 2))
        s.report_compute_done(current)
        self.assertEqual(s._busy['w1'], ahead)

    def test_ahead_interpolation_promotes_after_current_compute(self):
        s = FrameScheduler(['w1', 'w2'], prefetch=True, fused_ccsr=True)
        s.submit_interval(0)
        s.submit_interval(1)
        first = s.claim('w1')
        prefetched = s.claim_prefetch('w1')
        neighbor = s.claim('w2')
        s.complete(prefetched)
        s.report_prepared(first)
        s.report_prepared(neighbor)
        s.complete(s.claim_peer_transfer('w2'))
        s.report_compute_done(first)
        current = s.claim('w1')
        ahead = s.claim_ahead('w1')
        self.assertEqual((ahead.kind, ahead.frame), (Kind.INTERPOLATE, 0))
        self.assertIsNone(s.claim_ahead('w1'))
        s.report_compute_done(current)
        self.assertEqual(s._busy['w1'], ahead)
        s.report_compute_done(ahead, fused=True)
        s.complete_fused_interpolation(ahead)
        s.complete(current)
        s.complete(first)

    def test_seek_fences_ahead_work_until_drained(self):
        s = FrameScheduler(['w1', 'w2'], prefetch=True, fused_ccsr=True)
        s.submit_interval(0)
        s.submit_interval(1)
        first = s.claim('w1')
        prefetched = s.claim_prefetch('w1')
        neighbor = s.claim('w2')
        s.complete(prefetched)
        s.report_prepared(first)
        s.report_prepared(neighbor)
        s.complete(s.claim_peer_transfer('w2'))
        s.report_compute_done(first)
        current = s.claim('w1')
        ahead = s.claim_ahead('w1')
        s.reset()
        s.submit_interval(20)
        self.assertIsNone(s.claim('w1'))
        for command in (first, current, ahead, neighbor):
            s.acknowledge_drained(command)
        self.assertIsNotNone(s.claim('w1'))

    def test_ahead_promotes_on_final_completion_without_overlap(self):
        s = FrameScheduler(['w1', 'w2'], interp_mult=3,
                           prefetch=True, fused_ccsr=True)
        s.submit_interval(0)
        s.submit_interval(1)
        first = s.claim('w1')
        prefetch = s.claim_prefetch('w1')
        neighbor = s.claim('w2')
        s.complete(prefetch)
        s.report_prepared(first)
        s.report_prepared(neighbor)
        s.complete(s.claim_peer_transfer('w2'))
        s.complete(first)
        current = s.claim('w1')
        ahead = s.claim_ahead('w1')
        self.assertEqual((ahead.kind, ahead.frame), (Kind.INTERPOLATE, 0))
        s.complete(current)
        self.assertEqual(s._busy['w1'], ahead)
        s.complete(ahead)
        self.assertTrue(s._frames[0].interpolated)

    def test_ahead_sr_phases_are_queued_once_and_promoted(self):
        s = FrameScheduler(['w1', 'w2'], interp_mult=4, prefetch=True,
                           fused_ccsr=True, frame_slots=3)
        s.submit_interval(0)
        current = s.claim('w1')
        frame = s._frames[0]
        frame.prepared = frame.sr_done = frame.interpolated = True
        s.complete(current)
        first = s.claim('w1')
        self.assertEqual((first.kind, first.phase), (Kind.SUPER_RESOLVE_INTERPOLATED, 1))
        second = s.claim_ahead('w1')
        self.assertEqual((second.kind, second.phase), (Kind.SUPER_RESOLVE_INTERPOLATED, 2))
        self.assertIsNone(s.claim_ahead('w1'))
        s.complete(first)
        self.assertEqual(s._busy['w1'], second)
        third = s.claim_ahead('w1')
        self.assertEqual((third.kind, third.phase), (Kind.SUPER_RESOLVE_INTERPOLATED, 3))
        s.complete(second)
        s.complete(third)
        self.assertEqual(frame.interp_sr_done, {1, 2, 3})

    def test_source_precedes_interpolated_sr_on_idle_and_ahead_lanes(self):
        for ahead in (False, True):
            s = FrameScheduler(['w1', 'w2', 'w3'], interp_mult=4,
                               prefetch=True, fused_ccsr=True, frame_slots=3)
            s.submit_interval(0)
            first, neighbor = s.claim('w1'), s.claim('w2')
            s.complete(first)
            s.complete(neighbor)
            s._frames[0].interpolated = True
            current = s.claim('w1') if ahead else None
            if current:
                self.assertEqual(current.kind, Kind.SUPER_RESOLVE_INTERPOLATED)
            s.submit_interval(1)
            source = s.claim_ahead('w1') if ahead else s.claim('w1')
            self.assertEqual((source.kind, source.frame), (Kind.CCSR, 2))
            if current:
                s.complete(current)
                self.assertEqual(s._busy['w1'], source)

    def test_dynamic_ownership_and_neighbor_transfer(self):
        s = FrameScheduler(["w1", "w2", "w3"])
        s.submit_interval(0)
        s.submit_interval(1)
        a, b, c = (s.claim(w) for w in ("w1", "w2", "w3"))
        self.assertEqual([(x.kind, x.frame) for x in (a, b, c)],
                         [(Kind.PREPARE, 0), (Kind.PREPARE, 1), (Kind.PREPARE, 2)])
        for x in (a, b, c):
            s.complete(x)
        # SR on both owners can run before transfers or interpolation.
        a, b, c = (s.claim(w) for w in ("w1", "w2", "w3"))
        self.assertTrue(all(x.kind is Kind.SUPER_RESOLVE for x in (a, b, c)))
        transfer = s.claim_peer_transfer("w2")
        self.assertEqual((transfer.kind, transfer.frame, transfer.peer),
                         (Kind.TRANSFER, 1, "w1"))
        s.complete(b)
        output = s.claim_host_output("w2")
        self.assertEqual(output.kind, Kind.OUTPUT_FRAME)
        s.complete(transfer)
        s.complete(output)
        s.complete(a)
        interp = s.claim("w1")
        self.assertEqual((interp.kind, interp.frame), (Kind.INTERPOLATE, 0))
        s.complete(interp)
        s.complete(c)
        transfer = s.claim_peer_transfer("w3")
        self.assertEqual((transfer.kind, transfer.frame, transfer.peer),
                         (Kind.TRANSFER, 2, "w2"))
        s.complete(transfer)
        interp = s.claim("w2")
        self.assertEqual((interp.kind, interp.frame), (Kind.INTERPOLATE, 1))

    def test_seek_rejects_stale_completion(self):
        s = FrameScheduler(["w1", "w2"])
        s.submit_interval(8)
        old = s.claim("w1")
        s.reset()
        s.submit_interval(8)
        self.assertIsNone(s.claim("w1"))
        self.assertFalse(s.complete(old))
        current = s.claim("w1")
        self.assertTrue(s.complete(current))

    def test_same_worker_reuses_neighbor_without_transfer(self):
        s = FrameScheduler(["w1", "w2"])
        s.submit_interval(0)
        for expected in (Kind.PREPARE, Kind.SUPER_RESOLVE,
                         Kind.PREPARE, Kind.SUPER_RESOLVE, Kind.INTERPOLATE,
                         Kind.SUPER_RESOLVE_INTERPOLATED):
            cmd = s.claim("w1")
            self.assertEqual(cmd.kind, expected)
            s.complete(cmd)
        self.assertIsNone(s.claim("w1"))

    def test_validation_and_busy_claim(self):
        with self.assertRaises(ValueError):
            FrameScheduler(["same", "same"])
        s = FrameScheduler(["w1", "w2"])
        with self.assertRaises(ValueError):
            s.submit_interval(-1)
        s.submit_interval(0)
        cmd = s.claim("w1")
        with self.assertRaises(RuntimeError):
            s.claim("w1")
        s.complete(cmd)
        with self.assertRaises(ValueError):
            s.complete(cmd)

    def test_failure_invalidates_all_inflight_work(self):
        s = FrameScheduler(["w1", "w2"])
        s.submit_interval(0)
        first, second = s.claim("w1"), s.claim("w2")
        self.assertTrue(s.fail(first))
        self.assertFalse(s.complete(second))
        self.assertFalse(s.fail(first))
        s.submit_interval(0)
        self.assertIsNone(s.claim("w2"))
        self.assertTrue(s.acknowledge_drained(first))
        self.assertEqual(s.claim("w2").frame, 0)

    def test_all_three_lanes_can_overlap(self):
        s = FrameScheduler(["w1", "w2"])
        s.submit_interval(0)
        a, b = s.claim("w1"), s.claim("w2")
        s.complete(a)
        s.complete(b)
        sr0, sr1 = s.claim("w1"), s.claim("w2")
        s.complete(sr1)
        output = s.claim_host_output("w2")
        transfer = s.claim_peer_transfer("w2")
        self.assertEqual((sr0.kind, output.kind, transfer.kind),
                         (Kind.SUPER_RESOLVE, Kind.OUTPUT_FRAME, Kind.TRANSFER))
        # A peer transfer can finish while the receiving worker is still
        # computing, but interpolation waits for its source SR.
        s.complete(transfer)
        with self.assertRaises(RuntimeError):
            s.claim("w1")
        s.complete(sr0)
        interp = s.claim("w1")
        self.assertEqual(interp.kind, Kind.INTERPOLATE)
        s.complete(interp)
        self.assertEqual(s.claim("w1").kind, Kind.SUPER_RESOLVE_INTERPOLATED)
        # Output acknowledgement is independent of both computations.
        self.assertTrue(s.complete(output))

    def test_fourfold_interpolation_publishes_each_phase(self):
        s = FrameScheduler(["w1", "w2"], interp_mult=4)
        s.submit_interval(0)
        first, second = s.claim("w1"), s.claim("w2")
        s.complete(first)
        s.complete(second)
        transfer = s.claim_peer_transfer("w2")
        sr = s.claim("w1")
        s.complete(transfer)
        s.complete(sr)
        output = s.claim_host_output("w1")
        s.complete(output)
        interp = s.claim("w1")
        self.assertEqual(interp.kind, Kind.INTERPOLATE)
        s.complete(interp)
        for phase in (1, 2, 3):
            sr_interp = s.claim("w1")
            self.assertEqual((sr_interp.kind, sr_interp.phase),
                             (Kind.SUPER_RESOLVE_INTERPOLATED, phase))
            s.complete(sr_interp)
            result = s.claim_host_output("w1")
            self.assertEqual((result.kind, result.phase),
                             (Kind.OUTPUT_INTERPOLATED, phase))
            s.complete(result)

    def test_no_interpolation_does_not_send_neighbors(self):
        s = FrameScheduler(["w1", "w2"], interp_mult=1)
        s.submit_interval(0)
        for worker in ("w1", "w2"):
            command = s.claim(worker)
            s.complete(command)
            self.assertIsNone(s.claim_peer_transfer(worker))

    def test_prefetch_uses_two_nonadjacent_slots(self):
        s = FrameScheduler(["w1", "w2"], prefetch=True)
        s.submit_interval(0)
        current = s.claim("w1")
        self.assertEqual((current.kind, current.frame), (Kind.PREPARE, 0))
        # Only frame 1 remains: adjacent frames cannot occupy A/B slots.
        self.assertIsNone(s.claim_prefetch("w1"))
        s.submit_interval(1)
        future = s.claim_prefetch("w1")
        self.assertEqual((future.kind, future.frame), (Kind.PREFETCH, 2))
        s.complete(future)
        s.complete(current)
        sr = s.claim("w1")
        self.assertEqual((sr.kind, sr.frame), (Kind.SUPER_RESOLVE, 0))
        # Slots 0 and 2 remain occupied even after the transfer finished.
        s.submit_interval(2)
        self.assertIsNone(s.claim_prefetch("w1"))
        with self.assertRaises(RuntimeError):
            s.release_frame("w1", 0)
        s.complete(sr)
        prepare_b = s.claim("w1")
        self.assertEqual((prepare_b.kind, prepare_b.frame), (Kind.PREPARE, 2))
        self.assertTrue(prepare_b.prefetched)
        self.assertIsNone(s.claim_prefetch("w1"))

    def test_ready_prefetch_precedes_interpolation(self):
        s = FrameScheduler(["w1", "w2"], prefetch=True, fused_ccsr=True)
        s.submit_interval(0)
        s.submit_interval(1)
        first = s.claim("w1")
        future = s.claim_prefetch("w1")
        neighbor = s.claim("w2")
        s.complete(future)
        s.report_prepared(first)
        s.report_prepared(neighbor)
        transfer = s.claim_peer_transfer("w2")
        s.complete(transfer)
        s.report_compute_done(first)
        command = s.claim("w1")
        self.assertEqual((command.kind, command.frame, command.prefetched),
                         (Kind.CCSR, 2, True))

    def test_pending_prefetch_does_not_block_ready_interpolation(self):
        s = FrameScheduler(["w1", "w2"], prefetch=True, fused_ccsr=True)
        s.submit_interval(0)
        s.submit_interval(1)
        first = s.claim("w1")
        s.claim_prefetch("w1")
        neighbor = s.claim("w2")
        s.report_prepared(first)
        s.report_prepared(neighbor)
        transfer = s.claim_peer_transfer("w2")
        s.complete(transfer)
        s.report_compute_done(first)
        command = s.claim("w1")
        self.assertEqual((command.kind, command.frame), (Kind.INTERPOLATE, 0))

    def test_prefetch_reserves_separate_source_and_neighbor_cache_slots(self):
        s = FrameScheduler(["w1", "w2"], prefetch=True, fused_ccsr=True)
        s.submit_interval(0)
        s.submit_interval(1)
        first = s.claim("w1")
        future = s.claim_prefetch("w1")
        neighbor = s.claim("w2")
        self.assertEqual((first.frame, future.frame, neighbor.frame), (0, 2, 1))
        self.assertEqual(s.cache_slots("w1", 0), (0, 1))
        self.assertEqual(s.cache_slots("w1", 2), (2, 3))
        self.assertEqual(s.cache_slots("w2", 1), (0, 1))
        self.assertEqual(s.interpolation_slots("w1", 0), (4,))
        self.assertEqual(s.interpolation_slots("w1", 2), (7,))
        s.report_prepared(neighbor)
        transfer = s.claim_peer_transfer("w2")
        self.assertEqual(s.transfer_slots(transfer), (0, 1))
        s.complete(transfer)
        with self.assertRaises(ValueError):
            s.transfer_slots(transfer)

    def test_cache_slots_are_reused_only_after_release(self):
        s = FrameScheduler(["w1", "w2"], interp_mult=1,
                           prefetch=True, fused_ccsr=True)
        s.submit_interval(0)
        s.submit_interval(1)
        s.submit_interval(2)
        first = s.claim("w1")
        future = s.claim_prefetch("w1")
        s.complete(first)
        s.complete(future)
        output = s.claim_host_output("w1")
        s.complete(output)
        s.release_frame("w1", 0)
        with self.assertRaises(ValueError):
            s.cache_slots("w1", 0)
        next_compute = s.claim("w1")
        self.assertEqual(next_compute.frame, 2)
        s.submit_interval(3)
        later = s.claim_prefetch("w1")
        self.assertEqual(s.cache_slots("w1", later.frame), (0, 1))

    def test_fused_ccsr_sends_features_while_super_resolution_runs(self):
        s = FrameScheduler(["w1", "w2"], prefetch=True, fused_ccsr=True)
        s.submit_interval(0)
        s.submit_interval(1)
        first, second = s.claim("w1"), s.claim("w2")
        self.assertEqual((first.kind, second.kind), (Kind.CCSR, Kind.CCSR))
        self.assertTrue(s.report_prepared(first))
        self.assertTrue(s.report_prepared(second))
        transfer = s.claim_peer_transfer("w2")
        self.assertEqual((transfer.frame, transfer.peer), (1, "w1"))
        future = s.claim_prefetch("w1")
        self.assertEqual((future.kind, future.frame), (Kind.PREFETCH, 2))
        s.complete(transfer)
        with self.assertRaises(RuntimeError):
            s.claim("w1")
        s.complete(first)
        real_output = s.claim_host_output("w1")
        self.assertEqual(real_output.kind, Kind.OUTPUT_FRAME)
        interpolation = s.claim("w1")
        self.assertEqual((interpolation.kind, interpolation.frame),
                         (Kind.INTERPOLATE, 0))
        # Transfer, output and next compute are independent lanes.
        self.assertTrue(s.complete(real_output))
        self.assertTrue(s.complete(future))

    def test_compute_can_advance_while_previous_output_is_in_transit(self):
        s = FrameScheduler(["w1", "w2"], prefetch=True, fused_ccsr=True)
        s.submit_interval(0)
        first, second = s.claim("w1"), s.claim("w2")
        s.report_prepared(first)
        s.report_prepared(second)
        transfer = s.claim_peer_transfer("w2")
        s.complete(transfer)
        s.report_compute_done(first)
        interp = s.claim("w1")
        self.assertEqual(interp.kind, Kind.INTERPOLATE)
        # The real-frame output has not reached the host yet.
        self.assertIsNone(s.claim_host_output("w1"))
        s.report_compute_done(interp, fused=True)
        self.assertIsNone(s.claim_host_output("w1"))
        s.complete(first)
        output = s.claim_host_output("w1")
        self.assertEqual(output.kind, Kind.OUTPUT_FRAME)
        s.complete(output)
        s.complete_fused_interpolation(interp)
        output = s.claim_host_output("w1")
        self.assertEqual(output.kind, Kind.OUTPUT_INTERPOLATED)

    def test_prefetch_seek_and_failure_invalidate_reservations(self):
        s = FrameScheduler(["w1", "w2"], prefetch=True)
        s.submit_interval(0)
        s.submit_interval(1)
        current = s.claim("w1")
        future = s.claim_prefetch("w1")
        self.assertTrue(s.fail(future))
        self.assertFalse(s.complete(current))
        self.assertTrue(s.acknowledge_drained(future))
        s.submit_interval(5)
        s.submit_interval(6)
        current = s.claim("w1")
        future = s.claim_prefetch("w1")
        s.reset()
        self.assertFalse(s.complete(future))
        self.assertFalse(s.complete(current))

    def test_prefetch_slot_is_reused_only_after_output(self):
        s = FrameScheduler(["w1", "w2"], interp_mult=1, prefetch=True)
        s.submit_interval(0)
        s.submit_interval(1)
        s.submit_interval(2)
        current = s.claim("w1")
        future = s.claim_prefetch("w1")
        self.assertEqual(future.frame, 2)
        s.complete(future)
        s.complete(current)
        sr = s.claim("w1")
        s.complete(sr)
        with self.assertRaises(RuntimeError):
            s.release_frame("w1", 0)
        output = s.claim_host_output("w1")
        s.complete(output)
        s.release_frame("w1", 0)
        next_compute = s.claim("w1")
        self.assertEqual(next_compute.frame, 2)
        # Frame 1 is the earliest free frame, but it is adjacent to 2.
        # Frame 3 is also adjacent, so no B-slot reservation is valid.
        self.assertIsNone(s.claim_prefetch("w1"))
        s.submit_interval(3)
        later = s.claim_prefetch("w1")
        self.assertEqual(later.frame, 4)

    def test_prefetch_drains_with_nonadjacent_slots_and_uneven_workers(self):
        for seed in range(20):
            rng = random.Random(seed)
            workers = ["w1", "w2", "w3"]
            s = FrameScheduler(workers, prefetch=True)
            last = 12
            for frame in range(last):
                s.submit_interval(frame)
            running = {}
            outputs = []
            for _ in range(3000):
                for worker in workers:
                    for lane, claim in (("compute", s.claim),
                                        ("prefetch", s.claim_prefetch),
                                        ("peer", s.claim_peer_transfer),
                                        ("output", s.claim_host_output)):
                        key = (worker, lane)
                        if key in running:
                            continue
                        command = claim(worker)
                        if command is not None:
                            running[key] = command
                    held = sorted(s._slots[worker])
                    self.assertLessEqual(len(held), 2)
                    if len(held) == 2:
                        self.assertGreater(held[1] - held[0], 1)
                if not running:
                    break
                # Bias the selection to emulate unequal worker speeds.
                keys = list(running)
                chosen = rng.choice(keys + [key for key in keys if key[0] == "w1"])
                command = running.pop(chosen)
                s.complete(command)
                if command.kind in (Kind.OUTPUT_FRAME, Kind.OUTPUT_INTERPOLATED):
                    outputs.append((command.kind, command.frame))
                for worker in workers:
                    for frame in list(s._slots[worker]):
                        try:
                            s.release_frame(worker, frame, final=frame == last)
                        except RuntimeError:
                            pass
            else:
                self.fail("prefetch scheduler did not drain")
            self.assertEqual(len([x for x in outputs if x[0] is Kind.OUTPUT_FRAME]), last + 1)
            self.assertEqual(len([x for x in outputs if x[0] is Kind.OUTPUT_INTERPOLATED]), last)
            self.assertFalse(any(s._slots.values()))

    def test_fused_prefetch_and_early_transfer_drain_all_phases(self):
        for mult in (2, 3, 4):
            for seed in range(8):
                rng = random.Random(seed)
                workers = ["w1", "w2"]
                s = FrameScheduler(workers, interp_mult=mult,
                                   prefetch=True, fused_ccsr=True)
                last = 10
                for frame in range(last):
                    s.submit_interval(frame)
                running = {}
                outputs = set()
                for _ in range(3000):
                    for worker in workers:
                        for lane, claim in (("compute", s.claim),
                                            ("prefetch", s.claim_prefetch),
                                            ("peer", s.claim_peer_transfer),
                                            ("output", s.claim_host_output)):
                            key = (worker, lane)
                            if key not in running:
                                command = claim(worker)
                                if command is not None:
                                    running[key] = command
                    if not running:
                        break
                    for command in list(running.values()):
                        if command.kind is Kind.CCSR and rng.random() < 0.7:
                            s.report_prepared(command)
                    key = rng.choice(list(running))
                    command = running.pop(key)
                    s.complete(command)
                    if command.kind in (Kind.OUTPUT_FRAME, Kind.OUTPUT_INTERPOLATED):
                        result = (command.kind, command.frame, command.phase)
                        self.assertNotIn(result, outputs)
                        outputs.add(result)
                    for worker in workers:
                        held = sorted(s._slots[worker])
                        if len(held) == 2:
                            self.assertGreater(held[1] - held[0], 1)
                        for frame in held:
                            try:
                                s.release_frame(worker, frame, final=frame == last)
                            except RuntimeError:
                                pass
                else:
                    self.fail("fused prefetch scheduler did not drain")
                self.assertEqual(len(outputs), (last + 1) + last * (mult - 1))
                self.assertFalse(any(s._slots.values()))

    def test_three_frame_slots_drain_with_uneven_workers(self):
        for worker_count in (2, 3, 4):
            for mult in (2, 3, 4):
                for seed in range(12):
                    rng = random.Random(seed)
                    workers = [f"w{i}" for i in range(worker_count)]
                    s = FrameScheduler(workers, interp_mult=mult,
                                       prefetch=True, fused_ccsr=True,
                                       frame_slots=3)
                    last = 15
                    for frame in range(last):
                        s.submit_interval(frame)
                    running, outputs = {}, set()
                    for _ in range(5000):
                        for worker in workers:
                            for lane, claim in (("compute", s.claim),
                                                ("prefetch", s.claim_prefetch),
                                                ("peer", s.claim_peer_transfer),
                                                ("output", s.claim_host_output)):
                                key = (worker, lane)
                                if key not in running:
                                    command = claim(worker)
                                    if command is not None:
                                        running[key] = command
                        if not running:
                            break
                        for command in tuple(running.values()):
                            if command.kind is Kind.CCSR and rng.random() < 0.7:
                                s.report_prepared(command)
                        command = running.pop(rng.choice(list(running)))
                        s.complete(command)
                        if command.kind in (Kind.OUTPUT_FRAME, Kind.OUTPUT_INTERPOLATED):
                            result = (command.kind, command.frame, command.phase)
                            self.assertNotIn(result, outputs)
                            outputs.add(result)
                        for worker in workers:
                            held = sorted(s._slots[worker])
                            self.assertLessEqual(len(held), 3)
                            self.assertTrue(all(b - a > 1 for a, b in zip(held, held[1:])))
                            for frame in held:
                                try:
                                    s.release_frame(worker, frame, final=frame == last)
                                except RuntimeError:
                                    pass
                    else:
                        self.fail("three-slot scheduler did not drain")
                    self.assertEqual(len(outputs), (last + 1) + last * (mult - 1))
                    self.assertFalse(any(s._slots.values()))

    def test_three_frame_cache_regions_do_not_overlap(self):
        s = FrameScheduler(["w1", "w2"], interp_mult=4, prefetch=True,
                           fused_ccsr=True, frame_slots=3)
        for frame in range(5):
            s.submit_interval(frame)
        for i in range(3):
            command = s.claim("w1")
            s.complete(command)
            if i < 2:
                s.complete(s.claim("w2"))
        held = sorted(s._slots["w1"])
        self.assertEqual(held, [0, 2, 4])
        regions = []
        for frame in held:
            regions.extend(s.cache_slots("w1", frame))
            regions.extend(s.interpolation_slots("w1", frame))
        self.assertEqual(len(regions), len(set(regions)))
        self.assertLess(max(regions), 32)

    def test_out_of_order_worker_completion(self):
        for seed in range(30):
            rng = random.Random(seed)
            workers = ["w1", "w2", "w3", "w4"]
            s = FrameScheduler(workers)
            for frame in range(12):
                s.submit_interval(frame)
            running = {}
            completed = []
            for _ in range(1000):
                for worker in workers:
                    for lane, claim in (("compute", s.claim),
                                        ("peer", s.claim_peer_transfer),
                                        ("output", s.claim_host_output)):
                        key = (worker, lane)
                        if key in running:
                            continue
                        command = claim(worker)
                        if command is not None:
                            running[key] = command
                if not running:
                    break
                key = rng.choice(list(running))
                command = running.pop(key)
                s.complete(command)
                completed.append(command)
            else:
                self.fail("scheduler did not drain")
            for kind, expected in ((Kind.PREPARE, 13),
                                   (Kind.SUPER_RESOLVE, 13),
                                   (Kind.INTERPOLATE, 12),
                                   (Kind.SUPER_RESOLVE_INTERPOLATED, 12),
                                   (Kind.OUTPUT_FRAME, 13),
                                   (Kind.OUTPUT_INTERPOLATED, 12)):
                matching = [c.frame for c in completed if c.kind is kind]
                self.assertEqual(len(matching), expected)
                self.assertEqual(len(set(matching)), expected)


class OutputReorderTest(unittest.TestCase):
    def test_reorders_phases_and_invalidates_seek(self):
        out = OutputReorder(3, max_pending=2)
        self.assertTrue(out.submit(0, 0, 2, "c"))
        self.assertTrue(out.submit(0, 0, 1, "b"))
        with self.assertRaises(BufferError):
            out.submit(0, 1, 0, "d")
        self.assertIsNone(out.pop_ready())
        # The missing frame is accepted even if later frames fill the buffer.
        self.assertTrue(out.submit(0, 0, 0, "a"))
        self.assertEqual([out.pop_ready(), out.pop_ready(), out.pop_ready()],
                         [(0, 0, "a"), (0, 1, "b"), (0, 2, "c")])
        generation = out.reset(start_frame=5)
        self.assertFalse(out.submit(0, 5, 0, "stale"))
        self.assertTrue(out.submit(generation, 5, 0, "new"))
        with self.assertRaises(ValueError):
            out.submit(generation, 5, 0, "duplicate")
        self.assertEqual(out.pop_ready(), (5, 0, "new"))


if __name__ == "__main__":
    unittest.main()
