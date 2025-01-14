import unittest
import deltapv as dpv
from jax import numpy as jnp
import numpy as np
from scipy.optimize import minimize
from optimize import psc
from optimize import multi


class TestDeltaPV(unittest.TestCase):
    def test_iv(self):
        L = 3e-4
        J = 5e-6
        material = dpv.create_material(Chi=3.9,
                                       Eg=1.5,
                                       eps=9.4,
                                       Nc=8e17,
                                       Nv=1.8e19,
                                       mn=100,
                                       mp=100,
                                       Et=0,
                                       tn=1e-8,
                                       tp=1e-8,
                                       A=1e4)
        design = dpv.make_design(n_points=500,
                                 Ls=[J, L - J],
                                 mats=[material, material],
                                 Ns=[1e17, -1e15],
                                 Snl=1e7,
                                 Snr=0,
                                 Spl=0,
                                 Spr=1e7)
        results = dpv.simulate(design)
        v, j = results["iv"]

        v_correct = [
            0.0, 0.05, 0.1, 0.15000000000000002, 0.2, 0.25,
            0.30000000000000004, 0.35000000000000003, 0.4, 0.45, 0.5, 0.55,
            0.6000000000000001, 0.6500000000000001, 0.7000000000000001, 0.75,
            0.8, 0.85, 0.9, 0.9500000000000001
        ]

        j_correct = [
            0.01882799450659129, 0.018753370994746384, 0.018675073222852775,
            0.018592788678882418, 0.01850616015841796, 0.018414776404918568,
            0.018318159501526814, 0.01821574845029824, 0.018106874755825324,
            0.0179907188741479, 0.017866203205496447, 0.017731661626627034,
            0.017583825887487907, 0.01741498506998538, 0.017204823904941775,
            0.01689387681804267, 0.01628556057166174, 0.014630769395991339,
            0.008610345709349041, -0.018267911703588706
        ]

        self.assertTrue(jnp.allclose(v, v_correct), "Voltages do not match!")
        self.assertTrue(jnp.allclose(j, j_correct), "Currents do not match!")

    def test_psc(self):
        bounds = [(1, 5), (1, 5), (1, 20), (17, 20), (17, 20), (0, 3), (0, 3),
                  (1, 5), (1, 5), (1, 20), (17, 20), (17, 20), (0, 3), (0, 3),
                  (17, 20), (17, 20), (0, None)]

        x_init = np.array([
            1.661788237392516, 4.698293002285373, 19.6342803183675,
            18.83471869026531, 19.54569869328745, 0.7252792557586427,
            1.6231392299175988, 2.5268524699070234, 2.51936429069554,
            6.933634938056497, 19.41835918276137, 18.271793488422656,
            0.46319949214386513, 0.2058139980642224, 18.63975340175838,
            17.643726318153238
        ])

        opt = dpv.util.StatefulOptimizer(x_init=x_init,
                                         convr=psc.x2des,
                                         constr=psc.g,
                                         bounds=bounds)

        _ = opt.optimize(niters=5)
        y = opt.get_growth()
        y_correct = [
            -0.06472217283866989, -0.06472217283981127, -0.07843614940788636,
            -0.07960816056099762, -0.07960816056154761, -0.08121815838178086,
            -0.08280692825028514, -0.08439442795188701, -0.0859804446389292,
            -0.08756472922121954, -0.08914699356554232, -0.09072690784756043,
            -0.0923040979729528, -0.0938781430034618, -0.09544857258729693,
            -0.09701486442661211, -0.09857644191255212, -0.10013267211869231,
            -0.10168286448808203, -0.10322627078909115, -0.10322627078849755,
            -0.10482236380194764, -0.10641456502271636, -0.10800255883623565,
            -0.10958601798089881, -0.11116460584749599, -0.11273798022202182,
            -0.11430579936572852, -0.11586773200958977, -0.1174234742740333,
            -0.11897277971888988, -0.12051551628232086, -0.12205178467633127,
            -0.12358220015572646, -0.12510872485553753, -0.12663843054181192,
            -0.12821042714808645, -0.12965415083292625, -0.13108174718471985,
            -0.13249155138562257, -0.13388150801751528, -0.1352490143975222,
            -0.13659069545947036, -0.1379020806728254, -0.13917714113632051,
            -0.14040762802113582, -0.14158213059123698, -0.14268474126160166,
            -0.1436931759042192, -0.1445761486378758, -0.14528974259978986,
            -0.14577245267532185, -0.14593850212067958, -0.14566894070035807,
            -0.1456689407003483, -0.15183280702799856, -0.153947660993235
        ]
        self.assertTrue(np.allclose(y, y_correct),
                        "Objective growth does not match!")

    def test_multi(self):
        slsqp_res = minimize(multi.f_np,
                             x0=np.array([2.0, 1.2]),
                             method="SLSQP",
                             jac=True,
                             bounds=[(1.0, 3.0), (0.5, 2.0)],
                             options={
                                 "maxiter": 50,
                                 "disp": True
                             })
        x_correct = [2.20436169, 1.00000301]
        fun_correct = 1.3971387383125255e-8
        self.assertTrue(np.allclose(slsqp_res.x, x_correct), "Optimizer does not match!")
        self.assertTrue(np.allclose(slsqp_res.fun, fun_correct), "Minimum does not match!")


if __name__ == '__main__':
    unittest.main()
