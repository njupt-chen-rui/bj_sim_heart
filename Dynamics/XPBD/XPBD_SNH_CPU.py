import time

import taichi as ti
import numpy as np
import taichi.math as tm
from data.LV1 import meshData
from Geometry.body import Body


@ti.data_oriented
class XPBD_SNH_CPU_old:
    """
    use XPBD with Stable Neo-Hookean Materials to simulate
    """
    def __init__(self, body: Body, LameLa=16443.0, LameMu=336.0, dt=1./60., numSubsteps=1, numPosIters=10):
        self.body = body
        self.dt = dt
        self.numSubsteps = numSubsteps
        self.h = self.dt / self.numSubsteps
        self.numPosIters = numPosIters
        self.friction = 1000.0
        self.num_vertex = self.body.num_vertex
        self.num_element = self.body.num_tet
        self.pos = self.body.vertex
        self.prevPos = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.dx = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.f_ext = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.elements = self.body.elements
        self.fiber = self.body.tet_fiber
        self.vel = self.body.vel
        self.f_ext = ti.Vector.field(3, float, shape=(self.num_vertex,))
        # Lagrange_multiplier num = num of constraint != num of elem
        self.Lagrange_multiplier = ti.field(float, shape=(self.num_element, 3))
        self.mass = ti.field(float, shape=(self.num_vertex,))
        self.invMass = ti.field(float, shape=(self.num_vertex,))
        self.vol = self.body.volume
        self.invVol = ti.field(float, shape=(self.num_element,))
        self.Ds = ti.Matrix.field(3, 3, float, shape=(self.num_element,))
        self.F = ti.Matrix.field(3, 3, float, shape=(self.num_element,))
        self.grads = ti.Vector.field(3, float, shape=(4,))
        self.LameLa = LameLa
        self.LameMu = LameMu
        self.invLa = 1.0 / self.LameLa
        self.invMu = 1.0 / self.LameMu
        self.invh2 = 1.0 / self.h / self.h
        self.tet_Ta = body.tet_Ta
        self.max_tet_set = body.num_tet_set
        self.init()

    @ti.kernel
    def init(self):
        for i in self.pos:
            self.mass[i] = 0.0
            self.f_ext[i] = tm.vec3(0, 0, 0)
            self.vel[i] = tm.vec3(0, 0., 0)
            self.prevPos[i] = self.pos[i]

        for i in self.elements:
            self.invVol[i] = 1. / self.vol[i]
            pm = self.vol[i] / 4.0 * self.body.density
            for j in ti.static(range(4)):
                self.mass[self.elements[i][j]] += pm

        for i in self.pos:
            self.invMass[i] = 1.0 / self.mass[i]

    def update(self):
        for _ in range(self.numSubsteps):
            self.sub_step()

    def sub_step(self):
        self.preSolve()
        self.solve_Gauss_Seidel()
        self.postSolve()



    ### ------------------------------------------------------------------------------------------------------ ###

    @ti.kernel
    def preSolve(self):
        for i in self.pos:
            self.prevPos[i] = self.pos[i]
            self.vel[i] += self.h * self.f_ext[i] / self.mass[i]
            self.pos[i] += self.h * self.vel[i]
            if self.pos[i][1] < 0.0:
                self.pos[i] = self.prevPos[i]
                self.pos[i][1] = 0.0

        for i in self.elements:
            for j in ti.static(range(3)):
                self.Lagrange_multiplier[i, j] = 0.0

    def solve_Jacobi(self):
        for _ in range(self.numPosIters):
            self.solve_elem_Jacobi()

    def solve_Gauss_Seidel(self):
        for _ in range(self.numPosIters):
            self.solve_elem_Gauss_Seidel()

    @ti.kernel
    def solve_elem_Jacobi(self):
        pos, Ds, DmInv, F, grads, tet = ti.static(self.pos, self.Ds, self.body.DmInv, self.F, self.grads, self.elements)
        for i in self.pos:
            self.dx[i] = tm.vec3(0, 0., 0)

        for i in self.elements:
            """
            constraint = sqrt(tr(F^T@F))
            """
            Ds[i][0, 0] = pos[tet[i][1]][0] - pos[tet[i][0]][0]
            Ds[i][1, 0] = pos[tet[i][1]][1] - pos[tet[i][0]][1]
            Ds[i][2, 0] = pos[tet[i][1]][2] - pos[tet[i][0]][2]
            Ds[i][0, 1] = pos[tet[i][2]][0] - pos[tet[i][0]][0]
            Ds[i][1, 1] = pos[tet[i][2]][1] - pos[tet[i][0]][1]
            Ds[i][2, 1] = pos[tet[i][2]][2] - pos[tet[i][0]][2]
            Ds[i][0, 2] = pos[tet[i][3]][0] - pos[tet[i][0]][0]
            Ds[i][1, 2] = pos[tet[i][3]][1] - pos[tet[i][0]][1]
            Ds[i][2, 2] = pos[tet[i][3]][2] - pos[tet[i][0]][2]
            F[i] = Ds[i] @ DmInv[i]
            if i == 10:
                print(F[i])

            # constraint = sqrt(tr(F^T@F))
            constraint = tm.sqrt(F[i][0, 0] * F[i][0, 0] + F[i][0, 1] * F[i][0, 1] + F[i][0, 2] * F[i][0, 2]
                               + F[i][1, 0] * F[i][1, 0] + F[i][1, 1] * F[i][1, 1] + F[i][1, 2] * F[i][1, 2]
                               + F[i][2, 0] * F[i][2, 0] + F[i][2, 1] * F[i][2, 1] + F[i][2, 2] * F[i][2, 2])
            constraint_inv = 1.0 / constraint
            grads[0] = tm.vec3(0., 0., 0.)
            grads[1] = tm.vec3(0., 0., 0.)
            grads[2] = tm.vec3(0., 0., 0.)
            grads[3] = tm.vec3(0., 0., 0.)
            vecF = [tm.vec3(F[i][0, 0], F[i][1, 0], F[i][2, 0]),
                    tm.vec3(F[i][0, 1], F[i][1, 1], F[i][2, 1]),
                    tm.vec3(F[i][0, 2], F[i][1, 2], F[i][2, 2])]
            # grads[1] += constraint_inv * DmInv[i][0, 0] * vecF[0]
            # grads[1] += constraint_inv * DmInv[i][0, 1] * vecF[1]
            # grads[1] += constraint_inv * DmInv[i][0, 2] * vecF[2]
            # grads[2] += constraint_inv * DmInv[i][1, 0] * vecF[0]
            # grads[2] += constraint_inv * DmInv[i][1, 1] * vecF[1]
            # grads[2] += constraint_inv * DmInv[i][1, 2] * vecF[2]
            # grads[3] += constraint_inv * DmInv[i][2, 0] * vecF[0]
            # grads[3] += constraint_inv * DmInv[i][2, 1] * vecF[1]
            # grads[3] += constraint_inv * DmInv[i][2, 2] * vecF[2]
            # grads[0] = grads[0] - grads[1] - grads[2] - grads[3]
            grads[1] = constraint_inv * DmInv[i] @ vecF[0]
            grads[2] = constraint_inv * DmInv[i] @ vecF[1]
            grads[3] = constraint_inv * DmInv[i] @ vecF[2]
            grads[0][0] = 0.0 - grads[1][0] - grads[1][1] - grads[1][2]
            grads[0][1] = 0.0 - grads[2][0] - grads[2][1] - grads[2][2]
            grads[0][2] = 0.0 - grads[3][0] - grads[3][1] - grads[3][2]

            alpha1 = self.invMu * self.invVol[i] * self.invh2
            w_g = 0.0
            for j in ti.static(range(4)):
                w_g += self.invMass[self.elements[i][j]] * (grads[j].norm() ** 2)
            dlambda = (0. - constraint - alpha1 * self.Lagrange_multiplier[i, 0]) / (w_g + alpha1)
            self.Lagrange_multiplier[i, 0] += dlambda
            # for j in ti.static(range(4)):
            #     self.dx[self.elements[i][j]] += self.invMass[self.elements[i][j]] * dlambda * grads[j]

            """
            constraint = def(F) - gamma
            """
            dCdF = [tm.vec3(0., 0., 0.), tm.vec3(0., 0., 0.), tm.vec3(0., 0., 0.)]
            dCdF[0] = tm.cross(vecF[1], vecF[2])
            dCdF[1] = tm.cross(vecF[2], vecF[0])
            dCdF[2] = tm.cross(vecF[0], vecF[1])

            grads[0] = tm.vec3(0., 0., 0.)
            grads[1] = tm.vec3(0., 0., 0.)
            grads[2] = tm.vec3(0., 0., 0.)
            grads[3] = tm.vec3(0., 0., 0.)
            # grads[1] += DmInv[i][0, 0] * dCdF[0]
            # grads[1] += DmInv[i][0, 1] * dCdF[1]
            # grads[1] += DmInv[i][0, 2] * dCdF[2]
            # grads[2] += DmInv[i][1, 0] * dCdF[0]
            # grads[2] += DmInv[i][1, 1] * dCdF[1]
            # grads[2] += DmInv[i][1, 2] * dCdF[2]
            # grads[3] += DmInv[i][2, 0] * dCdF[0]
            # grads[3] += DmInv[i][2, 1] * dCdF[1]
            # grads[3] += DmInv[i][2, 2] * dCdF[2]
            # grads[0] = grads[0] - grads[1] - grads[2] - grads[3]

            grads[1] = DmInv[i] @ dCdF[0]
            grads[2] = DmInv[i] @ dCdF[1]
            grads[3] = DmInv[i] @ dCdF[2]
            grads[0][0] = 0.0 - grads[1][0] - grads[1][1] - grads[1][2]
            grads[0][1] = 0.0 - grads[2][0] - grads[2][1] - grads[2][2]
            grads[0][2] = 0.0 - grads[3][0] - grads[3][1] - grads[3][2]

            # constraint = tm.determinant(F[i]) - 1.0 - self.LameMu * self.invLa
            constraint = tm.determinant(F[i]) - 1.0
            if constraint == 0.0:
                continue
            w_g = 0.0
            for j in ti.static(range(4)):
                w_g += self.invMass[self.elements[i][j]] * (grads[j].norm() ** 2)
            if w_g == 0.0:
                continue
            alpha2 = self.invLa * self.invVol[i] * self.invh2
            # dlambda = (0. - constraint - alpha2 * self.Lagrange_multiplier[i, 1]) / (w_g + alpha2)
            dlambda = (- constraint) / (w_g + alpha2)
            self.Lagrange_multiplier[i, 1] += dlambda
            for j in ti.static(range(4)):
                self.dx[self.elements[i][j]] += self.invMass[self.elements[i][j]] * dlambda * grads[j]

            """
            active force:
            constraint = sqrt(I_ff)
            """
            # if method == Gauss-Seidel
            # Ds[i] = tm.mat3(pos[id1] - pos[id0], pos[id2] - pos[id0], pos[id3] - pos[id0])
            # F[i] = Ds[i] @ DmInv[i]
            # vecF = [tm.vec3(F[i][0, 0], F[i][1, 0], F[i][2, 0]),
            #         tm.vec3(F[i][0, 1], F[i][1, 1], F[i][2, 1]),
            #         tm.vec3(F[i][0, 2], F[i][1, 2], F[i][2, 2])]

            # f0 = self.fiber[i]
            # f = F[i] @ f0
            # I_ff = f.dot(f)
            # constraint = tm.sqrt(I_ff)
            # inv_sqrt_Iff = 1.0 / constraint
            # dIffdF = F[i] @ (f0.outer_product(f0))
            # vecdIffdF = [tm.vec3(dIffdF[0, 0], dIffdF[1, 0], dIffdF[2, 0]),
            #              tm.vec3(dIffdF[0, 1], dIffdF[1, 1], dIffdF[2, 1]),
            #              tm.vec3(dIffdF[0, 2], dIffdF[1, 2], dIffdF[2, 2])]
            # grads[0] = tm.vec3(0., 0., 0.)
            # grads[1] = tm.vec3(0., 0., 0.)
            # grads[2] = tm.vec3(0., 0., 0.)
            # grads[3] = tm.vec3(0., 0., 0.)
            # grads[1] += inv_sqrt_Iff * DmInv[i][0, 0] * vecdIffdF[0]
            # grads[1] += inv_sqrt_Iff * DmInv[i][0, 1] * vecdIffdF[1]
            # grads[1] += inv_sqrt_Iff * DmInv[i][0, 2] * vecdIffdF[2]
            # grads[2] += inv_sqrt_Iff * DmInv[i][1, 0] * vecdIffdF[0]
            # grads[2] += inv_sqrt_Iff * DmInv[i][1, 1] * vecdIffdF[1]
            # grads[2] += inv_sqrt_Iff * DmInv[i][1, 2] * vecdIffdF[2]
            # grads[3] += inv_sqrt_Iff * DmInv[i][2, 0] * vecdIffdF[0]
            # grads[3] += inv_sqrt_Iff * DmInv[i][2, 1] * vecdIffdF[1]
            # grads[3] += inv_sqrt_Iff * DmInv[i][2, 2] * vecdIffdF[2]
            # grads[0] = grads[0] - grads[1] - grads[2] - grads[3]
            #
            # w_g = 0.0
            # for j in ti.static(range(4)):
            #     w_g += self.invMass[self.elements[i][j]] * (grads[j].norm() ** 2)
            # alpha3 = self.invVol[i] * self.invh2 / self.tet_Ta[i]
            # dlambda = (0. - constraint - alpha3 * self.Lagrange_multiplier[i, 2]) / (w_g + alpha3)
            # self.Lagrange_multiplier[i, 2] += dlambda
            # for j in ti.static(range(4)):
            #     self.dx[self.elements[i][j]] += self.invMass[self.elements[i][j]] * dlambda * grads[j]

        for i in self.pos:
            self.pos[i] += self.dx[i]

        # for i in pos:
        #     if self.dx[i].norm() != 0.0:
        #         print(i, self.dx[i])

    @ti.kernel
    def solve_elem_Gauss_Seidel(self):
        pos, Ds, DmInv, F, grads, tet = ti.static(self.pos, self.Ds, self.body.DmInv, self.F, self.grads, self.elements)
        tet_set = ti.static(self.body.tet_set)
        # SNH
        for i in self.elements:
            """
            constraint = sqrt(tr(F^T@F))
            """
            id0, id1, id2, id3 = self.elements[i][0], self.elements[i][1], self.elements[i][1], self.elements[i][2]
            Ds[i][0, 0] = pos[tet[i][1]][0] - pos[tet[i][0]][0]
            Ds[i][1, 0] = pos[tet[i][1]][1] - pos[tet[i][0]][1]
            Ds[i][2, 0] = pos[tet[i][1]][2] - pos[tet[i][0]][2]
            Ds[i][0, 1] = pos[tet[i][2]][0] - pos[tet[i][0]][0]
            Ds[i][1, 1] = pos[tet[i][2]][1] - pos[tet[i][0]][1]
            Ds[i][2, 1] = pos[tet[i][2]][2] - pos[tet[i][0]][2]
            Ds[i][0, 2] = pos[tet[i][3]][0] - pos[tet[i][0]][0]
            Ds[i][1, 2] = pos[tet[i][3]][1] - pos[tet[i][0]][1]
            Ds[i][2, 2] = pos[tet[i][3]][2] - pos[tet[i][0]][2]
            F[i] = Ds[i] @ DmInv[i]
            # constraint = sqrt(tr(F^T@F))
            constraint = tm.sqrt(F[i][0, 0] * F[i][0, 0] + F[i][0, 1] * F[i][0, 1] + F[i][0, 2] * F[i][0, 2]
                               + F[i][1, 0] * F[i][1, 0] + F[i][1, 1] * F[i][1, 1] + F[i][1, 2] * F[i][1, 2]
                               + F[i][2, 0] * F[i][2, 0] + F[i][2, 1] * F[i][2, 1] + F[i][2, 2] * F[i][2, 2])
            eps = 1e-12
            constraint_inv = 1.0 / (constraint + eps)
            grads[0] = tm.vec3(0., 0., 0.)
            grads[1] = tm.vec3(0., 0., 0.)
            grads[2] = tm.vec3(0., 0., 0.)
            grads[3] = tm.vec3(0., 0., 0.)
            vecF = [tm.vec3(F[i][0, 0], F[i][1, 0], F[i][2, 0]),
                    tm.vec3(F[i][0, 1], F[i][1, 1], F[i][2, 1]),
                    tm.vec3(F[i][0, 2], F[i][1, 2], F[i][2, 2])]
            grads[1] += constraint_inv * DmInv[i][0, 0] * vecF[0]
            grads[1] += constraint_inv * DmInv[i][0, 1] * vecF[1]
            grads[1] += constraint_inv * DmInv[i][0, 2] * vecF[2]
            grads[2] += constraint_inv * DmInv[i][1, 0] * vecF[0]
            grads[2] += constraint_inv * DmInv[i][1, 1] * vecF[1]
            grads[2] += constraint_inv * DmInv[i][1, 2] * vecF[2]
            grads[3] += constraint_inv * DmInv[i][2, 0] * vecF[0]
            grads[3] += constraint_inv * DmInv[i][2, 1] * vecF[1]
            grads[3] += constraint_inv * DmInv[i][2, 2] * vecF[2]
            grads[0] = grads[0] - grads[1] - grads[2] - grads[3]
            alpha1 = self.invMu * self.invVol[i] * self.invh2
            w_g = 0.0
            for j in ti.static(range(4)):
                w_g += self.invMass[self.elements[i][j]] * (grads[j].norm() ** 2)
            dlambda = (0. - constraint - alpha1 * self.Lagrange_multiplier[i, 0]) / (w_g + alpha1)
            self.Lagrange_multiplier[i, 0] += dlambda
            for j in ti.static(range(4)):
                index = self.elements[i][j]
                self.dx[index] = self.invMass[index] * dlambda * grads[j]
                self.pos[index] += self.dx[index]

            """
            constraint = def(F) - gamma
            """

            Ds[i] = tm.mat3(pos[id1] - pos[id0], pos[id2] - pos[id0], pos[id3] - pos[id0])
            F[i] = Ds[i] @ DmInv[i]
            vecF[0] = tm.vec3(F[i][0, 0], F[i][1, 0], F[i][2, 0])
            vecF[1] = tm.vec3(F[i][0, 1], F[i][1, 1], F[i][2, 1])
            vecF[2] = tm.vec3(F[i][0, 2], F[i][1, 2], F[i][2, 2])
            # vecF = [tm.vec3(F[i][0, 0], F[i][1, 0], F[i][2, 0]),
            #         tm.vec3(F[i][0, 1], F[i][1, 1], F[i][2, 1]),
            #         tm.vec3(F[i][0, 2], F[i][1, 2], F[i][2, 2])]

            dCdF = [tm.vec3(0., 0., 0.), tm.vec3(0., 0., 0.), tm.vec3(0., 0., 0.)]
            dCdF[0] = tm.cross(vecF[1], vecF[2])
            dCdF[1] = tm.cross(vecF[2], vecF[0])
            dCdF[2] = tm.cross(vecF[0], vecF[1])

            grads[0] = tm.vec3(0., 0., 0.)
            grads[1] = tm.vec3(0., 0., 0.)
            grads[2] = tm.vec3(0., 0., 0.)
            grads[3] = tm.vec3(0., 0., 0.)
            grads[1] += DmInv[i][0, 0] * dCdF[0]
            grads[1] += DmInv[i][0, 1] * dCdF[1]
            grads[1] += DmInv[i][0, 2] * dCdF[2]
            grads[2] += DmInv[i][1, 0] * dCdF[0]
            grads[2] += DmInv[i][1, 1] * dCdF[1]
            grads[2] += DmInv[i][1, 2] * dCdF[2]
            grads[3] += DmInv[i][2, 0] * dCdF[0]
            grads[3] += DmInv[i][2, 1] * dCdF[1]
            grads[3] += DmInv[i][2, 2] * dCdF[2]
            grads[0] = grads[0] - grads[1] - grads[2] - grads[3]
            constraint = tm.determinant(F[i]) - 1.0 - self.LameMu * self.invLa
            w_g = 0.0
            for j in ti.static(range(4)):
                w_g += self.invMass[self.elements[i][j]] * (grads[j].norm() ** 2)
            alpha2 = self.invLa * self.invVol[i] * self.invh2
            dlambda = (0. - constraint - alpha2 * self.Lagrange_multiplier[i, 1]) / (w_g + alpha2)
            self.Lagrange_multiplier[i, 1] += dlambda
            for j in ti.static(range(4)):
                index = self.elements[i][j]
                self.dx[index] = self.invMass[index] * dlambda * grads[j]
                self.pos[index] += self.dx[index]

            """
            active force:
            constraint = sqrt(I_ff)
            """
            # if method == Gauss-Seidel
            # Ds[i] = tm.mat3(pos[id1] - pos[id0], pos[id2] - pos[id0], pos[id3] - pos[id0])
            # F[i] = Ds[i] @ DmInv[i]
            # vecF = [tm.vec3(F[i][0, 0], F[i][1, 0], F[i][2, 0]),
            #         tm.vec3(F[i][0, 1], F[i][1, 1], F[i][2, 1]),
            #         tm.vec3(F[i][0, 2], F[i][1, 2], F[i][2, 2])]

            # f0 = self.fiber[i]
            # f = F[i] @ f0
            # I_ff = f.dot(f)
            # constraint = tm.sqrt(I_ff)
            # inv_sqrt_Iff = 1.0 / constraint
            # dIffdF = F[i] @ (f0.outer_product(f0))
            # vecdIffdF = [tm.vec3(dIffdF[0, 0], dIffdF[1, 0], dIffdF[2, 0]),
            #              tm.vec3(dIffdF[0, 1], dIffdF[1, 1], dIffdF[2, 1]),
            #              tm.vec3(dIffdF[0, 2], dIffdF[1, 2], dIffdF[2, 2])]
            # grads[0] = tm.vec3(0., 0., 0.)
            # grads[1] = tm.vec3(0., 0., 0.)
            # grads[2] = tm.vec3(0., 0., 0.)
            # grads[3] = tm.vec3(0., 0., 0.)
            # grads[1] += inv_sqrt_Iff * DmInv[i][0, 0] * vecdIffdF[0]
            # grads[1] += inv_sqrt_Iff * DmInv[i][0, 1] * vecdIffdF[1]
            # grads[1] += inv_sqrt_Iff * DmInv[i][0, 2] * vecdIffdF[2]
            # grads[2] += inv_sqrt_Iff * DmInv[i][1, 0] * vecdIffdF[0]
            # grads[2] += inv_sqrt_Iff * DmInv[i][1, 1] * vecdIffdF[1]
            # grads[2] += inv_sqrt_Iff * DmInv[i][1, 2] * vecdIffdF[2]
            # grads[3] += inv_sqrt_Iff * DmInv[i][2, 0] * vecdIffdF[0]
            # grads[3] += inv_sqrt_Iff * DmInv[i][2, 1] * vecdIffdF[1]
            # grads[3] += inv_sqrt_Iff * DmInv[i][2, 2] * vecdIffdF[2]
            # grads[0] = grads[0] - grads[1] - grads[2] - grads[3]
            #
            # w_g = 0.0
            # for j in ti.static(range(4)):
            #     w_g += self.invMass[self.elements[i][j]] * (grads[j].norm() ** 2)
            # alpha3 = self.invVol[i] * self.invh2 / self.tet_Ta[i]
            # dlambda = (0. - constraint - alpha3 * self.Lagrange_multiplier[i, 2]) / (w_g + alpha3)
            # self.Lagrange_multiplier[i, 2] += dlambda
            # for j in ti.static(range(4)):
            #     self.dx[self.elements[i][j]] += self.invMass[self.elements[i][j]] * dlambda * grads[j]

    @ti.kernel
    def postSolve(self):
        for i in self.pos:
            self.vel[i] = (self.pos[i] - self.prevPos[i]) / self.h

    def sub_step_Jacobi(self):
        self.preSolve()
        self.solve_Jacobi()
        self.postSolve()

    def sub_step_Gauss_Seidel(self):
        self.preSolve()
        self.solve_Gauss_Seidel()
        self.postSolve()

    def update_Jacobi(self):
        for _ in range(self.numSubsteps):
            self.sub_step_Jacobi()

    def update_Gauss_Seidel(self):
        for _ in range(self.numSubsteps):
            self.sub_step_Gauss_Seidel()


@ti.data_oriented
class XPBD_SNH_CPU:
    """
    use XPBD with Stable Neo-Hookean Materials to simulate
    """
    def __init__(self, body: Body, LameLa=16443.0, LameMu=336.0, dt=1./60., numSubsteps=1, numPosIters=1):
        self.body = body
        self.dt = dt
        self.numSubsteps = numSubsteps
        self.h = self.dt / self.numSubsteps
        self.numPosIters = numPosIters
        self.numPosIters_Jacobi = self.numPosIters * 20
        self.friction = 1000.0
        self.num_vertex = self.body.num_vertex
        self.num_element = self.body.num_tet
        self.pos = self.body.vertex
        self.prevPos = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.dx = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.f_ext = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.elements = self.body.elements
        self.fiber = self.body.tet_fiber
        self.vel = self.body.vel
        self.f_ext = ti.Vector.field(3, float, shape=(self.num_vertex,))
        # Lagrange_multiplier num = num of constraint != num of elem
        self.Lagrange_multiplier = ti.field(float, shape=(self.num_element, 3))
        self.mass = ti.field(float, shape=(self.num_vertex,))
        self.invMass = ti.field(float, shape=(self.num_vertex,))
        self.vol = self.body.volume
        self.invVol = ti.field(float, shape=(self.num_element,))
        self.Ds = ti.Matrix.field(3, 3, float, shape=(self.num_element,))
        self.F = ti.Matrix.field(3, 3, float, shape=(self.num_element,))
        self.grads = ti.Vector.field(3, float, shape=(4,))
        self.dpos = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.LameLa = LameLa
        self.LameMu = LameMu
        self.invLa = 1.0 / self.LameLa
        self.invMu = 1.0 / self.LameMu
        self.invh2 = 1.0 / self.h / self.h
        self.tet_Ta = body.tet_Ta
        self.max_tet_set = body.num_tet_set
        self.init()

    @ti.kernel
    def init(self):
        for i in self.pos:
            self.mass[i] = 0.0
            self.f_ext[i] = tm.vec3(0, 0, 0)
            self.vel[i] = tm.vec3(0, 0., 0)
            self.prevPos[i] = self.pos[i]

        for i in self.elements:
            self.invVol[i] = 1. / self.vol[i]
            pm = self.vol[i] / 4.0 * self.body.density
            for j in ti.static(range(4)):
                self.mass[self.elements[i][j]] += pm

        for i in self.pos:
            self.invMass[i] = 1.0 / self.mass[i]

    def update(self):
        for _ in range(self.numSubsteps):
            self.sub_step()
        print("done")

    def update_Jacobi(self):
        for _ in range(self.numSubsteps):
            self.sub_step_Jacobi()

    def sub_step(self):
        self.preSolve()
        self.solve_Gauss_Seidel()
        # self.solve_Jacobi()
        self.postSolve()

    def sub_step_Jacobi(self):
        self.preSolve()
        # self.solve_Gauss_Seidel()
        self.solve_Jacobi()
        self.postSolve()

    @ti.kernel
    def preSolve(self):
        for i in self.pos:
            self.prevPos[i] = self.pos[i]
            self.vel[i] += self.h * self.f_ext[i] * self.invMass[i]
            self.pos[i] += self.h * self.vel[i]

    @ti.kernel
    def postSolve(self):
        for i in self.pos:
            if self.pos[i][1] < 0.0:
                self.pos[i][1] = 0.0
                v = self.prevPos[i] - self.pos[i]
                self.pos[i][0] += v[0] * ti.min(1.0, self.h * self.friction)
                self.pos[i][2] += v[2] * ti.min(1.0, self.h * self.friction)

        for i in self.pos:
            self.vel[i] = (self.pos[i] - self.prevPos[i]) / self.h

    def solve_Gauss_Seidel(self):
        for _ in range(self.numPosIters):
            self.solve_elem_Gauss_Seidel()

    def solve_Jacobi(self):
        for _ in range(self.numPosIters_Jacobi):
            self.solve_elem_Jacobi()

    def solve_elem_Gauss_Seidel(self):
        pos, vel, tet, ir, g = ti.static(self.pos, self.vel, self.elements, self.body.DmInv, self.grads)
        for i in range(self.num_element):
            C = 0.0
            devCompliance = 1.0 * self.invMu
            volCompliance = 1.0 * self.invLa

            # tr(F) = 3
            id = tm.ivec4(0, 0, 0, 0)
            for j in range(4):
                id[j] = tet[i][j]

            v1 = pos[id[1]] - pos[id[0]]
            v2 = pos[id[2]] - pos[id[0]]
            v3 = pos[id[3]] - pos[id[0]]
            Ds = tm.mat3(v1, v2, v3)
            Ds = Ds.transpose()
            F = Ds @ ir[i]
            F_col0 = tm.vec3(F[0, 0], F[1, 0], F[2, 0])
            F_col1 = tm.vec3(F[0, 1], F[1, 1], F[2, 1])
            F_col2 = tm.vec3(F[0, 2], F[1, 2], F[2, 2])
            r_s = tm.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]
                        + v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]
                        + v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2])
            r_s_inv = 1.0 / r_s
            g[1] = tm.vec3(0., 0., 0.)
            g[1] += F_col0 * (r_s_inv * ir[i][0, 0])
            g[1] += F_col1 * (r_s_inv * ir[i][0, 1])
            g[1] += F_col2 * (r_s_inv * ir[i][0, 2])

            g[2] = tm.vec3(0., 0., 0.)
            g[2] += F_col0 * (r_s_inv * ir[i][1, 0])
            g[2] += F_col1 * (r_s_inv * ir[i][1, 1])
            g[2] += F_col2 * (r_s_inv * ir[i][1, 2])

            g[3] = tm.vec3(0., 0., 0.)
            g[3] += F_col0 * (r_s_inv * ir[i][2, 0])
            g[3] += F_col1 * (r_s_inv * ir[i][2, 1])
            g[3] += F_col2 * (r_s_inv * ir[i][2, 2])

            C = r_s
            self.applyToElem(i, C, devCompliance)

            # det(F) = 1
            v1 = pos[id[1]] - pos[id[0]]
            v2 = pos[id[2]] - pos[id[0]]
            v3 = pos[id[3]] - pos[id[0]]
            Ds = tm.mat3(v1, v2, v3)
            Ds = Ds.transpose()
            F = Ds @ ir[i]
            F_col0 = tm.vec3(F[0, 0], F[1, 0], F[2, 0])
            F_col1 = tm.vec3(F[0, 1], F[1, 1], F[2, 1])
            F_col2 = tm.vec3(F[0, 2], F[1, 2], F[2, 2])
            dF0 = F_col1.cross(F_col2)
            dF1 = F_col2.cross(F_col0)
            dF2 = F_col0.cross(F_col1)

            g[1] = tm.vec3(0., 0., 0.)
            g[1] += dF0 * ir[i][0, 0]
            g[1] += dF1 * ir[i][0, 1]
            g[1] += dF2 * ir[i][0, 2]

            g[2] = tm.vec3(0., 0., 0.)
            g[2] += dF0 * ir[i][1, 0]
            g[2] += dF1 * ir[i][1, 1]
            g[2] += dF2 * ir[i][1, 2]

            g[3] = tm.vec3(0., 0., 0.)
            g[3] += dF0 * ir[i][2, 0]
            g[3] += dF1 * ir[i][2, 1]
            g[3] += dF2 * ir[i][2, 2]

            vol = self.mat3_determinant(F)
            C = vol - 1.0 - volCompliance / devCompliance
            self.applyToElem(i, C, volCompliance)

            # Iff = 1
            v1 = pos[id[1]] - pos[id[0]]
            v2 = pos[id[2]] - pos[id[0]]
            v3 = pos[id[3]] - pos[id[0]]
            Ds = tm.mat3(v1, v2, v3)
            Ds = Ds.transpose()
            F = Ds @ ir[i]
            f0 = self.body.tet_fiber[i]
            f = F @ f0
            C = tm.sqrt(f.dot(f))
            C_inv = 1.0 / C
            dIff = f0.outer_product(f0)
            dIff0 = tm.vec3(dIff[0, 0], dIff[1, 0], dIff[2, 0])
            dIff1 = tm.vec3(dIff[0, 1], dIff[1, 1], dIff[2, 1])
            dIff2 = tm.vec3(dIff[0, 2], dIff[1, 2], dIff[2, 2])

            g[1] = tm.vec3(0., 0., 0.)
            g[1] += dIff0 * (C_inv * ir[i][0, 0])
            g[1] += dIff1 * (C_inv * ir[i][0, 1])
            g[1] += dIff2 * (C_inv * ir[i][0, 2])

            g[2] = tm.vec3(0., 0., 0.)
            g[2] += dIff0 * (C_inv * ir[i][1, 0])
            g[2] += dIff1 * (C_inv * ir[i][1, 1])
            g[2] += dIff2 * (C_inv * ir[i][1, 2])

            g[3] = tm.vec3(0., 0., 0.)
            g[3] += dIff0 * (C_inv * ir[i][2, 0])
            g[3] += dIff1 * (C_inv * ir[i][2, 1])
            g[3] += dIff2 * (C_inv * ir[i][2, 2])

            self.applyToElem(i, C, 1.0 / self.body.tet_Ta[i])
            # if i == 0:
            #     print(f)
            #     print(C)
                # print(self.body.DmInv[i])
                # print(F)

    def applyToElem(self, elemNr, C, compliance):
        if C == 0.0:
            return
        g, pos, elem, h, invVol, invMass = ti.static(self.grads, self.pos, self.elements, self.h, self.invVol, self.invMass)
        g[0] = tm.vec3(0., 0., 0.)
        g[0] -= g[1]
        g[0] -= g[2]
        g[0] -= g[3]

        w = 0.0
        for i in range(4):
            id = elem[elemNr][i]
            w += (g[i][0] * g[i][0] + g[i][1] * g[i][1] + g[i][2] * g[i][2]) * invMass[id]

        if w == 0.0:
            return
        alpha = compliance / h / h * invVol[elemNr]
        dlambda = -C / (w + alpha)

        for i in range(4):
            id = elem[elemNr][i]
            pos[id] += g[i] * (dlambda * invMass[id])

    def mat3_determinant(self, mat):
        res = mat[0, 0] * (mat[1, 1] * mat[2, 2] - mat[2, 1] * mat[1, 2])\
              - mat[1, 0] * (mat[0, 1] * mat[2, 2] - mat[2, 1] * mat[0, 2])\
              + mat[2, 0] * (mat[0, 1] * mat[1, 2] - mat[1, 1] * mat[0, 2])
        return res

    @ti.kernel
    def solve_elem_Jacobi(self):
        pos, vel, tet, ir, g, dpos = ti.static(self.pos, self.vel, self.elements, self.body.DmInv, self.grads, self.dpos)
        for i in range(self.num_vertex):
            self.dpos[i] = tm.vec3(0.0, 0.0, 0.0)

        for i in range(self.num_element):
            C = 0.0
            devCompliance = 1.0 * self.invMu
            volCompliance = 1.0 * self.invLa

            # tr(F) = 3
            id = tm.ivec4(0, 0, 0, 0)
            for j in ti.static(range(4)):
                id[j] = tet[i][j]

            v1 = pos[id[1]] - pos[id[0]]
            v2 = pos[id[2]] - pos[id[0]]
            v3 = pos[id[3]] - pos[id[0]]
            Ds = tm.mat3(v1, v2, v3)
            Ds = Ds.transpose()
            F = Ds @ ir[i]
            F_col0 = tm.vec3(F[0, 0], F[1, 0], F[2, 0])
            F_col1 = tm.vec3(F[0, 1], F[1, 1], F[2, 1])
            F_col2 = tm.vec3(F[0, 2], F[1, 2], F[2, 2])
            r_s = tm.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]
                          + v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]
                          + v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2])
            r_s_inv = 1.0 / r_s
            g[1] = tm.vec3(0., 0., 0.)
            g[1] += F_col0 * (r_s_inv * ir[i][0, 0])
            g[1] += F_col1 * (r_s_inv * ir[i][0, 1])
            g[1] += F_col2 * (r_s_inv * ir[i][0, 2])

            g[2] = tm.vec3(0., 0., 0.)
            g[2] += F_col0 * (r_s_inv * ir[i][1, 0])
            g[2] += F_col1 * (r_s_inv * ir[i][1, 1])
            g[2] += F_col2 * (r_s_inv * ir[i][1, 2])

            g[3] = tm.vec3(0., 0., 0.)
            g[3] += F_col0 * (r_s_inv * ir[i][2, 0])
            g[3] += F_col1 * (r_s_inv * ir[i][2, 1])
            g[3] += F_col2 * (r_s_inv * ir[i][2, 2])

            C = r_s
            self.applyToElem_Jacobi(i, C, devCompliance)


            # det(F) = 1
            v1 = pos[id[1]] - pos[id[0]]
            v2 = pos[id[2]] - pos[id[0]]
            v3 = pos[id[3]] - pos[id[0]]
            Ds = tm.mat3(v1, v2, v3)
            Ds = Ds.transpose()
            F = Ds @ ir[i]
            F_col0 = tm.vec3(F[0, 0], F[1, 0], F[2, 0])
            F_col1 = tm.vec3(F[0, 1], F[1, 1], F[2, 1])
            F_col2 = tm.vec3(F[0, 2], F[1, 2], F[2, 2])
            dF0 = F_col1.cross(F_col2)
            dF1 = F_col2.cross(F_col0)
            dF2 = F_col0.cross(F_col1)

            g[1] = tm.vec3(0., 0., 0.)
            g[1] += dF0 * ir[i][0, 0]
            g[1] += dF1 * ir[i][0, 1]
            g[1] += dF2 * ir[i][0, 2]

            g[2] = tm.vec3(0., 0., 0.)
            g[2] += dF0 * ir[i][1, 0]
            g[2] += dF1 * ir[i][1, 1]
            g[2] += dF2 * ir[i][1, 2]

            g[3] = tm.vec3(0., 0., 0.)
            g[3] += dF0 * ir[i][2, 0]
            g[3] += dF1 * ir[i][2, 1]
            g[3] += dF2 * ir[i][2, 2]

            vol = F.determinant()
            C = vol - 1.0 - volCompliance / devCompliance
            self.applyToElem_Jacobi(i, C, volCompliance)

            # Iff = 1
            v1 = pos[id[1]] - pos[id[0]]
            v2 = pos[id[2]] - pos[id[0]]
            v3 = pos[id[3]] - pos[id[0]]
            Ds = tm.mat3(v1, v2, v3)
            Ds = Ds.transpose()
            F = Ds @ ir[i]
            f0 = self.body.tet_fiber[i]
            f = F @ f0
            C = tm.sqrt(f.dot(f))
            C_inv = 1.0 / C
            dIff = f0.outer_product(f0)
            dIff0 = tm.vec3(dIff[0, 0], dIff[1, 0], dIff[2, 0])
            dIff1 = tm.vec3(dIff[0, 1], dIff[1, 1], dIff[2, 1])
            dIff2 = tm.vec3(dIff[0, 2], dIff[1, 2], dIff[2, 2])

            g[1] = tm.vec3(0., 0., 0.)
            g[1] += dIff0 * (C_inv * ir[i][0, 0])
            g[1] += dIff1 * (C_inv * ir[i][0, 1])
            g[1] += dIff2 * (C_inv * ir[i][0, 2])

            g[2] = tm.vec3(0., 0., 0.)
            g[2] += dIff0 * (C_inv * ir[i][1, 0])
            g[2] += dIff1 * (C_inv * ir[i][1, 1])
            g[2] += dIff2 * (C_inv * ir[i][1, 2])

            g[3] = tm.vec3(0., 0., 0.)
            g[3] += dIff0 * (C_inv * ir[i][2, 0])
            g[3] += dIff1 * (C_inv * ir[i][2, 1])
            g[3] += dIff2 * (C_inv * ir[i][2, 2])

            self.applyToElem_Jacobi(i, C, 1.0 / self.body.tet_Ta[i])

        for i in range(self.num_vertex):
            self.pos[i] += self.dpos[i]

    @ti.func
    def applyToElem_Jacobi(self, elemNr, C, compliance):
        g, dpos, elem, h, invVol, invMass = ti.static(self.grads, self.dpos, self.elements, self.h, self.invVol, self.invMass)
        g[0] = tm.vec3(0., 0., 0.)
        g[0] -= g[1]
        g[0] -= g[2]
        g[0] -= g[3]

        w = 0.0
        for i in range(4):
            id = elem[elemNr][i]
            w += (g[i][0] * g[i][0] + g[i][1] * g[i][1] + g[i][2] * g[i][2]) * invMass[id]

        dlambda = 0.0
        if w == 0.0:
            dlambda = 0.0
        else:
            alpha = compliance / h / h * invVol[elemNr]
            dlambda = -C / (w + alpha)

        for i in range(4):
            id = elem[elemNr][i]
            dpos[id] += g[i] * (dlambda * invMass[id])


if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f32)
    # 顶点位置
    pos_np = np.array(meshData['verts'], dtype=float)
    pos_np = pos_np.reshape((-1, 3))
    # 四面体顶点索引
    tet_np = np.array(meshData['tetIds'], dtype=int)
    tet_np = tet_np.reshape((-1, 4))
    # edge
    edge_np = np.array(meshData['tetEdgeIds'], dtype=int)
    edge_np = edge_np.reshape((-1, 2))
    # surface tri index
    # surf_tri_np = np.array(meshData['tetSurfaceTriIds'], dtype=int)
    # surf_tri_np = surf_tri_np.reshape((-1, 3))
    # tet_fiber方向
    fiber_tet_np = np.array(meshData['fiberDirection'], dtype=float)
    fiber_tet_np = fiber_tet_np.reshape((-1, 3))
    # tet_sheet方向
    sheet_tet_np = np.array(meshData['sheetDirection'], dtype=float)
    sheet_tet_np = sheet_tet_np.reshape((-1, 3))
    # num_edge_set
    num_edge_set_np = np.array(meshData['num_edge_set'], dtype=int)[0]
    # edge_set
    edge_set_np = np.array(meshData['edge_set'], dtype=int)
    # num_tet_set
    num_tet_set_np = np.array(meshData['num_tet_set'], dtype=int)[0]
    # tet_set
    tet_set_np = np.array(meshData['tet_set'], dtype=int)
    # bou_tag
    bou_tag_dirichlet_np = np.array(meshData['bou_tag_dirichlet'], dtype=int)
    bou_tag_neumann_np = np.array(meshData['bou_tag_neumann'], dtype=int)

    body = Body(pos_np, tet_np, edge_np, fiber_tet_np, sheet_tet_np, num_edge_set_np, edge_set_np, num_tet_set_np,
                tet_set_np, bou_tag_dirichlet_np, bou_tag_neumann_np)
    # body.show()https://github.com/yuki-koyama/elasty.git
    body.translation(0.0, 20.5, 0.0)
    sys = XPBD_SNH_CPU(body=body)
    # print(sys.pos[10])
    # sys.update_Jacobi()
    # print(sys.pos[10])
    # start = time.time()
    # sys.update_Jacobi()
    # end = time.time()
    # print(end - start)

    #                                      gui                                     #
    # ---------------------------------------------------------------------------- #
    # set parameter
    windowLength = 1024
    lengthScale = min(windowLength, 512)
    light_distance = lengthScale / 25.

    x_min = min(body.vertex[i][0] for i in range(body.vertex.shape[0]))
    x_max = max(body.vertex[i][0] for i in range(body.vertex.shape[0]))
    y_min = min(body.vertex[i][1] for i in range(body.vertex.shape[0]))
    y_max = max(body.vertex[i][1] for i in range(body.vertex.shape[0]))
    z_min = min(body.vertex[i][2] for i in range(body.vertex.shape[0]))
    z_max = max(body.vertex[i][2] for i in range(body.vertex.shape[0]))
    center = np.array([(x_min + x_max) / 2., (y_min + y_max) / 2., (z_min + z_max) / 2.])

    # init the window, canvas, scene and camera
    window = ti.ui.Window("body show", (windowLength, windowLength), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # initial camera position
    camera.position(0.5, 50, 100)
    camera.lookat(0.5, 0.3, 0.5)
    camera.fov(55)

    while window.running:
        sys.update_Jacobi()
        # set the camera, you can move around by pressing 'wasdeq'
        camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.LMB)
        scene.set_camera(camera)

        # set the light
        scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
        scene.ambient_light(color=(0.5, 0.5, 0.5))

        # draw
        # scene.particles(pos, radius=0.02, color=(0, 1, 1))
        scene.mesh(body.vertex, indices=body.surfaces, color=(1.0, 0, 0), two_sided=False)

        # show the frame
        canvas.scene(scene)
        window.show()
