import unittest
import numpy as np
from audit import aggregate, counts, metrics


class AuditTests(unittest.TestCase):
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
