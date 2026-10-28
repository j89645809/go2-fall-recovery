import unittest
import torch
from baseline_observations import assemble

class ObservationTests(unittest.TestCase):
    def data(self):
        widths=dict(lin_vel=3,ang_vel=3,gravity=3,commands=3,joint_pos=12,
                    default_joint_pos=12,joint_vel=12,actions=12,push_force=2,
                    push_torque=3,friction=1,body_mass=1,limb_mass=18,measured_heights=187)
        d={k:torch.ones(2,v) for k,v in widths.items()}
        d['root_z']=torch.ones(2)
        return d
    def test_order_scales(self):
        actor,critic=assemble(**self.data())
        self.assertEqual(tuple(actor.shape),(2,45));self.assertEqual(tuple(critic.shape),(2,260))
        torch.testing.assert_close(actor[0,:9],torch.tensor([.25,.25,.25,1.,1.,1.,2.,2.,.25]))
        torch.testing.assert_close(critic[0,:6],torch.tensor([2.,2.,2.,.25,.25,.25]))
        torch.testing.assert_close(critic[:,55:73],torch.full((2,18),.2))
        torch.testing.assert_close(critic[:,73:],torch.full((2,187),-2.5))
    def test_missing_mass_rejected(self):
        d=self.data();d['limb_mass']=torch.ones(2,12)
        with self.assertRaises(ValueError):assemble(**d)

if __name__=='__main__':unittest.main()
