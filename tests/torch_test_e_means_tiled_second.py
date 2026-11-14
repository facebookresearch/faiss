# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file.
"""Complete tiled SECOND, valid candidates, tails and consumer lifetime."""
import unittest
import torch
from torch_test_e_means_cute import _ready


@unittest.skipUnless(_ready(), 'pinned CuTe stack on SM90 required')
class TestTiledCompleteSecond(unittest.TestCase):
    def test_closed_form_lanes_tiles_ties_and_nan_exclusion(self):
        from faiss.contrib.e_means.torch_backend import CuteBackend
        cases=((0,1,1.,False),(0,2,1.,False),(0,4,1.,False),(2,3,1.,False),
               (192,193,1.,False),(0,192,1.,False),(0,1,0.,False),(0,1,1.,True),
               (192,193,1.,True))
        for first_id,second_id,value,nan in cases:
            with self.subTest(first=first_id,second=second_id,tie=value==0,nan=nan):
                k,d,b=385,96,17
                c=torch.zeros(k,d,device='cuda');c[:,0]=100
                if nan:c[:,0]=torch.nan
                c[first_id,0]=0;c[second_id,0]=value
                x=torch.zeros(b,d,device='cuda',dtype=torch.bfloat16);x[:,0]=.25
                backend=CuteBackend(4096);p=backend.prepare_centroids(c,torch.bfloat16)[0]
                backing=torch.full((2*b+3,),-123.,device='cuda');distance=backing[1:b+1];second=backing[b+2:2*b+2]
                labels=torch.empty(b,device='cuda',dtype=torch.int64)
                actual,ids=backend.assign(x,p,distance_out=distance,labels_out=labels,second_out=second)
                self.assertIs(actual,distance);self.assertIs(ids,labels)
                self.assertEqual(p.bindings[(b,'second')].launcher.__name__,'launch_tiled_gemm')
                self.assertIsNone(p.bindings[(b,'second')].row_norms)
                torch.testing.assert_close(ids,torch.full_like(ids,first_id),rtol=0,atol=0)
                torch.testing.assert_close(distance,torch.full_like(distance,.0625),rtol=0,atol=0)
                truth=.0625 if value==0 else .5625
                torch.testing.assert_close(second,torch.full_like(second,truth),rtol=0,atol=0)
                complete=second.clone();backend.complete_second(x,ids,second)
                self.assertTrue(torch.equal(second,complete))
                self.assertEqual(backing[[0,b+1,-1]].tolist(),[-123.,-123.,-123.])
                top1,top1_ids=backend.assign(x,p)
                self.assertTrue(torch.equal(top1,distance));self.assertTrue(torch.equal(top1_ids,ids))

    def test_normalized_random_euclidean_and_spherical(self):
        from faiss.contrib.e_means.torch_backend import CuteBackend
        g=torch.Generator(device='cuda').manual_seed(20261114)
        c=torch.nn.functional.normalize(torch.randn(385,96,device='cuda',generator=g),dim=1)
        x=torch.nn.functional.normalize(torch.randn(17,96,device='cuda',generator=g),dim=1).bfloat16()
        for spherical in (False,True):
            with self.subTest(spherical=spherical):
                backend=CuteBackend(4096);p=backend.prepare_centroids(c,torch.bfloat16,spherical=spherical)[0]
                second=torch.empty(17,device='cuda');distance,labels=backend.assign(x,p,spherical=spherical,second_out=second)
                q,z=x.double(),p.centroids.double()
                truth=-2*(q[:,None]*z[None]).sum(2) if spherical else (q[:,None]-z[None]).square().sum(2)
                order=torch.argsort(truth,dim=1,stable=True)
                torch.testing.assert_close(labels,order[:,0],rtol=0,atol=0)
                torch.testing.assert_close(distance.double(),truth.gather(1,order[:,:1]).flatten(),rtol=0,atol=1e-5)
                torch.testing.assert_close(second.double(),truth.gather(1,order[:,1:2]).flatten(),rtol=0,atol=1e-5)

    def test_capture_refresh_and_complete_output_noop(self):
        from faiss.contrib.e_means.torch_backend import CuteBackend
        c=torch.zeros(385,96,device='cuda');c[:,0]=100;c[0,0]=0;c[1,0]=1
        x=torch.zeros(17,96,device='cuda',dtype=torch.bfloat16);x[:,0]=.25
        backend=CuteBackend(4096);p=backend.prepare_centroids(c,torch.bfloat16)[0]
        distance=torch.empty(17,device='cuda');second=torch.empty_like(distance);labels=torch.empty(17,device='cuda',dtype=torch.int64)
        backend.assign(x,p,distance_out=distance,labels_out=labels,second_out=second);torch.cuda.synchronize();backend.capture_state()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            p=backend.prepare_centroids(c,torch.bfloat16)[0]
            backend.assign(x,p,distance_out=distance,labels_out=labels,second_out=second)
            backend.complete_second(x,labels,second)
        pointers=tuple(v.data_ptr() for v in (distance,labels,second))
        for coordinate,other,winner,d1,d2 in ((.25,1.,0,.0625,.5625),(.75,1.,1,.0625,.5625),(.25,2.,0,.0625,3.0625),(.25,1.,0,.0625,.5625)):
            x[:,0]=coordinate;c[1,0]=other;graph.replay()
            torch.testing.assert_close(labels,torch.full_like(labels,winner),rtol=0,atol=0)
            torch.testing.assert_close(distance,torch.full_like(distance,d1),rtol=0,atol=0)
            torch.testing.assert_close(second,torch.full_like(second,d2),rtol=0,atol=0)
            self.assertEqual(pointers,tuple(v.data_ptr() for v in (distance,labels,second)))
        torch.cuda.synchronize();graph.reset()


if __name__=='__main__':unittest.main()
