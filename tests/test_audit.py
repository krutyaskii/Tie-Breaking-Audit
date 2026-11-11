import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from audit import aggregate, counts, metrics, collect, array_hash, load_run, run_lock


class AuditTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_and_rejects_changes(self):
        class Scorer:
            nentity = 5
            def score_candidates(self, triples, mode):
                return np.tile(np.array([3,2,2,1,0],dtype=np.float32),(len(triples),1))
        triples = np.array([[0,0,1],[2,0,3],[4,0,1]],dtype=np.int64)
        indexes = ({(0,1):{0,2,4}, (0,3):{2}}, {(0,0):{1},(2,0):{3},(4,0):{1}})
        config = {'query_count':6,'rows_per_mode':3,'nentity':5,'batch':2,'target_sha256':array_hash(np.r_[triples[:,0],triples[:,2]])}
        with tempfile.TemporaryDirectory() as tmp:
            full, resumed = Path(tmp)/'full',Path(tmp)/'resumed'
            self.assertTrue(collect(Scorer(),triples,indexes,full,config))
            self.assertFalse(collect(Scorer(),triples,indexes,resumed,config,stop_after_batches=1))
            with self.assertRaisesRegex(ValueError,'incompatible'):
                collect(Scorer(),triples,indexes,resumed,{**config,'batch':1},resume=True)
            self.assertTrue(collect(Scorer(),triples,indexes,resumed,config,resume=True))
            with np.load(full/'per_query.npz') as a, np.load(resumed/'per_query.npz') as b:
                for key in ('row','mode','target','B','T','retained'):
                    np.testing.assert_array_equal(a[key],b[key])
            self.assertEqual(load_run(full)[1],load_run(resumed)[1])
            meta = json.loads((full/'metadata.json').read_text())
            meta['config']['nentity'] = 6
            (full/'metadata.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                load_run(full)

    def test_union_filtering_both_modes(self):
        from rotate.common import truth_indexes
        splits = {'train':np.array([[0,0,4],[4,0,0]]),
                  'valid':np.array([[1,0,4],[4,0,1]]),
                  'test':np.array([[2,0,4],[4,0,2]])}
        heads,tails = truth_indexes(splits)
        scores = np.array([5,4,4,4,1],dtype=np.float32)
        self.assertEqual(heads[(0,4)],{0,1,2})
        self.assertEqual(tails[(4,0)],{0,1,2})
        for known in (heads[(0,4)],tails[(4,0)]):
            self.assertEqual(counts(scores,2,known), (0,2,3))

    def test_bruteforce_scores(self):
        rng = np.random.default_rng(912)
        for _ in range(50):
            scores = rng.integers(0,4,size=10).astype(np.float32)
            target = int(rng.integers(0,10))
            known = {0,3,7,target}
            kept = [i for i in range(10) if i not in known or i == target]
            better = sum(scores[i] > scores[target] for i in kept)
            tied = sum(scores[i] == scores[target] for i in kept)
            self.assertEqual(counts(scores,target,known),(better,tied,len(kept)))

    def test_seed_statistics_nonzero_ties(self):
        b,t = np.array([0,1,2,3]),np.array([4,4,4,4])
        arrays = {'row':np.array([0,1,0,1]),'mode':np.array([0,0,1,1]),'target':np.zeros(4,dtype=int),
                  'B':b,'T':t,'retained':np.full(4,10)}
        result = aggregate(arrays,{'query_count':4,'rows_per_mode':2,'nentity':10})
        values = []
        for seed in range(5):
            ranks = b+np.random.default_rng(seed).integers(1,t+1)
            values.append(sum(1/int(r) for r in ranks)/4)
        mean = sum(values)/5
        std = (sum((v-mean)**2 for v in values)/4)**.5
        self.assertAlmostEqual(result['seed_mean']['MRR'],mean)
        self.assertAlmostEqual(result['seed_std_ddof1']['MRR'],std)
        self.assertGreater(std,0)

    def test_exclusive_run_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with run_lock(Path(tmp)):
                with self.assertRaises(ValueError):
                    with run_lock(Path(tmp)):
                        pass
            with run_lock(Path(tmp)):
                pass

    def test_hand_cases(self):
        for scores, target, known, expected, rr in [
            ([.9,.8,.7],1,[],(1,1),1/2),
            ([.9,.8,.8,.1],1,[],(1,2),5/12),
            ([.5,.5,.5,.5],0,[],(0,4),25/48),
            ([.9,.8,.8,.1],1,[0,1],(0,2),3/4),
        ]:
            b,t,_ = counts(np.array(scores,dtype=np.float32),target,known)
            self.assertEqual((b,t),expected)
            self.assertAlmostEqual(metrics([b],[t])['expected_random']['MRR'],rr)

    def test_float32_equality_and_invalid(self):
        x = np.float32(.5)
        self.assertEqual(counts(np.array([x,np.nextafter(x,np.float32(1))]),0,[] )[:2],(1,1))
        for a in (np.array([np.nan],dtype=np.float32), np.array([1.],dtype=np.float64)):
            with self.assertRaises(ValueError): counts(a,0,[])
        with self.assertRaises(ValueError): counts(np.array([1.],dtype=np.float32),1,[])

    def test_direct_oracle(self):
        for b in [0,1,9,14000]:
            for t in [1,2,9,100]:
                m = metrics([b],[t])
                self.assertAlmostEqual(m['expected_random']['MRR'],sum(1/(b+i) for i in range(1,t+1))/t,places=12)
                for k in [1,3,10]:
                    self.assertAlmostEqual(m['expected_random'][f'Hits@{k}'],sum(b+i<=k for i in range(1,t+1))/t)

    def test_aggregation_and_order(self):
        a = {'row':np.array([0,1,0,1]), 'mode':np.array([0,0,1,1]), 'target':np.zeros(4,dtype=int), 'B':np.array([0,1,2,3]), 'T':np.ones(4,dtype=int), 'retained':np.full(4,10)}
        m = aggregate(a, {'query_count':4,'rows_per_mode':2,'nentity':10})
        self.assertEqual(m['overall']['gap'],0)
        self.assertEqual(m['seed_std_ddof1']['MRR'],0)
        a['row'][0] = 1
        with self.assertRaises(ValueError): aggregate(a, {'query_count':4,'rows_per_mode':2,'nentity':10})


if __name__ == '__main__': unittest.main()
