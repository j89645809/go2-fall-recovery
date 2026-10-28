import unittest
import torch
import baseline_math as m

class FormulaTests(unittest.TestCase):
    def test_orientation(self):
        g=torch.tensor([[0.,0.,-1.],[1.,0.,0.],[0.,0.,1.]])
        torch.testing.assert_close(m.orientation(g),torch.tensor([0.,2.,4.]))
        expected=torch.exp(-torch.tensor([0.,1.,4.])/(2*.15**2))
        torch.testing.assert_close(m.orientation_pos_gaussian(g),expected)

    def test_target_schedule_and_gate(self):
        g=torch.tensor([[0.,0.,-1.]])
        low=torch.tensor([[0.,1.5,-2.4]*4])
        high=torch.tensor([[-.05,.8,-1.5,.05,.8,-1.5,-.05,1.,-1.5,.05,1.,-1.5]])
        for it,q in [(0,low),(750,(low+high)/2),(1500,high),(2000,high)]:
            torch.testing.assert_close(m.step2_target_pos(q,g,it),torch.ones(1))
        torch.testing.assert_close(m.step2_target_pos(low,-g,0),torch.zeros(1))

    def test_relative_height(self):
        torch.testing.assert_close(m.base_height_pos(torch.tensor([1.45]),torch.ones(1,187)),torch.ones(1))

    def test_actions(self):
        a=torch.tensor([[1.,1.]])
        torch.testing.assert_close(m.action_smoothness_2(a,a,torch.zeros_like(a)),torch.zeros(1))
        torch.testing.assert_close(m.max_velocity(torch.tensor([[1.,-1.,0.]])),torch.tensor([.4]))

    def test_total_requires_complete_mapping(self):
        with self.assertRaises(ValueError): m.total_reward({},.02)
        terms={k:torch.zeros(1) for k in m.SCALES}
        terms['orientation']=torch.ones(1)
        torch.testing.assert_close(m.total_reward(terms,.02),torch.zeros(1))

if __name__=='__main__': unittest.main()
