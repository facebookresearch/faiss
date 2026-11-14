# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file.
"""BF16 row-owned second results, independent of empirical route selection."""
import unittest
import torch
from torch_test_e_means_cute import _ready


@unittest.skipUnless(_ready(), "pinned CuTe stack on SM90 required")
class TestRowOwnedSecond(unittest.TestCase):
    def fixture(self, b=17, k=128, second_id=1, second_value=1., first_id=0):
        from faiss.contrib.e_means import cute, cute_row_owned as row
        c=torch.zeros(k,96,device='cuda');c[:,0]=100;c[first_id,0]=0;c[second_id,0]=second_value
        x=cute._aligned_operand(b,96,c.device,torch.bfloat16);x.zero_();x[:,0]=.25
        prepared=cute._prepare_centroids(c)
        plan=row.row_owned_plan(b,k,96,c.device)
        self.assertIsNotNone(plan)
        return cute,row,x,prepared,plan

    def test_distinct_second_same_slot_other_tile_and_ties(self):
        cases=((17,128,0,1,1.),(129,256,0,128,1.),(17,128,0,1,0.),
               (17,128,0,2,1.),(17,128,0,4,1.),(17,128,0,6,1.),
               (17,128,0,10,1.),
               (17,128,2,3,1.),(17,128,4,5,1.))
        for b,k,first_id,j,value in cases:
            with self.subTest(b=b,k=k,first_id=first_id,second_id=j,tie=value==0):
                _,row,x,p,plan=self.fixture(b,k,j,value,first_id)
                first=torch.empty(b,device='cuda');labels=torch.empty(b,device='cuda',dtype=torch.int64)
                backing=torch.full((b+2,),-123.,device='cuda');second=backing[1:-1]
                top1=row.prepare_row_owned(x,p,first,labels,plan)
                row.launch_row_owned(top1);old=first.clone()
                top2=row.prepare_row_owned(x,p,first,labels,plan,second_scores=second)
                row.launch_row_owned(top2)
                self.assertTrue(torch.equal(first,old))
                torch.testing.assert_close(labels,torch.full_like(labels,first_id),rtol=0,atol=0)
                # Direct scores omit the row norm, exactly as the tiled epilogue.
                norm=x.float().square().sum(1)
                torch.testing.assert_close(first+norm,torch.full_like(first,.0625),rtol=0,atol=0)
                truth=.5625 if value else .0625
                torch.testing.assert_close(second+norm,torch.full_like(second,truth),rtol=0,atol=0)
                self.assertEqual(backing[[0,-1]].tolist(),[-123.,-123.])
                row.launch_row_owned(top1)
                self.assertTrue(torch.equal(first,old))

    def test_normalized_random_represented_fp64_oracle(self):
        from faiss.contrib.e_means import cute, cute_row_owned as row
        generator=torch.Generator(device='cuda').manual_seed(20261114)
        c=torch.nn.functional.normalize(torch.randn(128,96,device='cuda',generator=generator),dim=1)
        original=torch.nn.functional.normalize(torch.randn(17,96,device='cuda',generator=generator),dim=1)
        x=cute._aligned_operand(17,96,c.device,torch.bfloat16);x.copy_(original)
        for spherical in (False,True):
            with self.subTest(spherical=spherical):
                p=cute._prepare_centroids(c,spherical=spherical)
                first=torch.empty(17,device='cuda');second=torch.empty_like(first)
                labels=torch.empty(17,device='cuda',dtype=torch.int64)
                plan=row.row_owned_plan(17,128,96,c.device)
                bound=row.prepare_row_owned(x,p,first,labels,plan,second_scores=second)
                row.launch_row_owned(bound)
                q,centroids=x.double(),p.centroids.double()
                truth=-2*(q[:,None]*centroids[None]).sum(2) if spherical else (
                    q[:,None]-centroids[None]).square().sum(2)
                order=torch.argsort(truth,dim=1,stable=True)
                norm=0 if spherical else x.float().square().sum(1)
                torch.testing.assert_close(labels,order[:,0],rtol=0,atol=0)
                # Existing complete BF16 second-distance precision bound.
                torch.testing.assert_close((first+norm).double(),truth.gather(1,order[:,:1]).flatten(),rtol=0,atol=1e-5)
                torch.testing.assert_close((second+norm).double(),truth.gather(1,order[:,1:2]).flatten(),rtol=0,atol=1e-5)

    def test_prepared_second_capture_and_changed_rows(self):
        _,row,x,p,plan=self.fixture()
        first=torch.empty(len(x),device='cuda');second=torch.empty_like(first)
        labels=torch.empty(len(x),device='cuda',dtype=torch.int64)
        bound=row.prepare_row_owned(x,p,first,labels,plan,second_scores=second)
        row.launch_row_owned(bound);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):row.launch_row_owned(bound)
        pointers=tuple(t.data_ptr() for t in (first,labels,second))
        for value,label,raw_first,raw_second in ((.25,0,0.,.5),(.75,1,-.5,0.),(.25,0,0.,.5)):
            x[:,0]=value;graph.replay()
            torch.testing.assert_close(first,torch.full_like(first,raw_first),rtol=0,atol=0)
            torch.testing.assert_close(second,torch.full_like(second,raw_second),rtol=0,atol=0)
            torch.testing.assert_close(labels,torch.full_like(labels,label),rtol=0,atol=0)
            self.assertEqual(pointers,tuple(t.data_ptr() for t in (first,labels,second)))
        torch.cuda.synchronize();graph.reset()

    def test_natural_dispatch_row_owned_complete_outputs(self):
        cute,_,x,p,_=self.fixture()
        distance=torch.empty(len(x),device='cuda');second=torch.empty_like(distance)
        for second_requested in (False,True):
            with self.subTest(second=second_requested):
                optional={'second_out':second} if second_requested else {}
                actual,labels=cute._assign_rows(x,p,distance_out=distance,**optional)
                self.assertIs(actual,distance)
                key=(len(x),'second') if second_requested else len(x)
                self.assertEqual(p.bindings[key].launcher.__name__,'launch_row_owned')
                if not second_requested:
                    self.assertIsNone(p.bindings[key].second_scores)
                torch.testing.assert_close(labels,torch.zeros_like(labels),rtol=0,atol=0)
                torch.testing.assert_close(distance,torch.full_like(distance,.0625),rtol=0,atol=0)
                if second_requested:
                    torch.testing.assert_close(second,torch.full_like(second,.5625),rtol=0,atol=0)

    def test_natural_spherical_second_uses_native_score_units(self):
        from faiss.contrib.e_means import cute
        c=torch.zeros(128,96,device='cuda');c[:,1]=1
        c[0,:2]=torch.tensor([1.,0.],device='cuda')
        c[1,:2]=torch.tensor([.5,.75**.5],device='cuda')
        x=torch.zeros(17,96,device='cuda',dtype=torch.bfloat16);x[:,0]=1
        p=cute._prepare_centroids(c,spherical=True)
        second=torch.empty(17,device='cuda')
        distance,labels=cute._assign_rows(x,p,second_out=second)
        self.assertEqual(p.bindings[(17,'second')].launcher.__name__,'launch_row_owned')
        torch.testing.assert_close(labels,torch.zeros_like(labels),rtol=0,atol=0)
        torch.testing.assert_close(distance,torch.full_like(distance,-2.),rtol=0,atol=0)
        torch.testing.assert_close(second,torch.full_like(second,-1.),rtol=0,atol=0)

    def test_baseline_tiled_when_row_owned_exceeds_shared_memory(self):
        from faiss.contrib.e_means import cute, cute_row_owned as row
        from faiss.contrib.e_means.torch_backend import CuteBackend
        c=torch.zeros(128,256,device='cuda');c[:,0]=100;c[0,0]=0;c[1,0]=1
        x=torch.zeros(17,256,device='cuda',dtype=torch.bfloat16);x[:,0]=.25
        self.assertIsNone(row.row_owned_plan(17,128,256,x.device))
        backend=CuteBackend(4096)
        p=backend.prepare_centroids(c,torch.bfloat16)[0];second=torch.empty(17,device='cuda')
        distance,labels=cute._assign_rows(x,p,second_out=second)
        backend.complete_second(x,labels,second)
        self.assertEqual(p.bindings[(17,'second')].launcher.__name__,'launch_tiled_gemm')
        torch.testing.assert_close(labels,torch.zeros_like(labels),rtol=0,atol=0)
        torch.testing.assert_close(distance,torch.full_like(distance,.0625),rtol=0,atol=0)
        torch.testing.assert_close(second,torch.full_like(second,.5625),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
