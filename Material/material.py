import numpy as np
import taichi as ti
import taichi.math as tm


@ti.data_oriented
class NeoHookean:
    """
    elastic energy density:
    ψ = C1 * (I1 - 3 - 2 * ln(J)) + D1 * (J - 1)**2,
    σ = J^(-1) * ∂ψ/∂F * F^T = 2*C1*J^(-1)*(B - I) + 2*D1*(J-1)*I
    https://en.wikipedia.org/wiki/Neo-Hookean_solid
    """

    def __init__(self, Youngs_modulus, Poisson_ratio):
        self.Youngs_modulus = Youngs_modulus
        self.Poisson_ratio = Poisson_ratio
        self.LameLa = Youngs_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
        self.LameMu = Youngs_modulus / (2 * (1 + Poisson_ratio))
        self.C1 = self.LameMu / 2.
        self.D1 = self.LameLa / 2.

    @ti.kernel
    def constitutive_small_deform(self, deformationGradient: ti.template(),
                                  cauchy_stress: ti.template()):
        # C1, D1 = ti.static(self.C1, self.D1)
        mu, la = ti.static(self.LameMu, self.LameLa)
        identity3 = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        for i in ti.grouped(deformationGradient):
            F = deformationGradient[i]
            J = F.determinant()
            B = F @ F.transpose()  # left Cauchy-Green Strain tensor
            cauchy_stress[i] = mu / J * (B - identity3) + la * (J - 1.) * identity3

    @ti.kernel
    def constitutive_large_deform(self, deformationGradient: ti.template(),
                                  cauchy_stress: ti.template()):
        # C1, D1 = ti.static(self.C1, self.D1)
        mu, la = ti.static(self.LameMu, self.LameLa)
        identity3 = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        for i in ti.grouped(deformationGradient):
            F = deformationGradient[i]
            J = F.determinant()
            B = F @ F.transpose()  # left Cauchy-Green Strain tensor
            cauchy_stress[i] = mu / J * (B - identity3) + la * (J - 1.) * identity3

    @ti.func
    def elastic_energy_density(self, deformationGradient):
        F = deformationGradient
        J = F.determinant()
        B = F @ F.transpose()
        return self.C1 * (B.trace() - 3. - 2. * ti.log(J)) + self.D1 * (J - 1.) ** 2


@ti.data_oriented
class Stable_Neo_Hookean:
    """
    elastic energy density:
    I1 = tr(S), I2 = tr(F^T@F), I3 = det(F)
    ψ = μ/2 * (I2-3) - μ(I3-1) + λ/2 * (I3-1)^2
    """

    def __init__(self, Youngs_modulus, Poisson_ratio):
        self.Youngs_modulus = Youngs_modulus
        self.Poisson_ratio = Poisson_ratio
        self.LameLa = Youngs_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
        self.LameMu = Youngs_modulus / (2 * (1 + Poisson_ratio))

    @ti.func
    def ComputePsiDeriv(self, deformation_gradient: ti.template(), fiber_direction: ti.template()):
        """
        input deformationGradient F,
        return Energy density Psi and the first Piola-Kirchhoff tensor P
        """
        mu, la = ti.static(self.LameMu, self.LameLa)

        F = deformation_gradient
        J = F.determinant()

        # 修改反转元素
        U, sigma, V = ti.svd(F, ti.f32)
        if sigma[2, 2] < 0:
            sigma[2, 2] = -sigma[2, 2]

        # 定义不变量: I1 = tr(S), I2 = tr(F^T@F), I3 = det(F)
        I1 = sigma[0, 0] + sigma[1, 1] + sigma[2, 2]
        I2 = sigma[0, 0] * sigma[0, 0] + sigma[1, 1] * sigma[1, 1] + sigma[2, 2] * sigma[2, 2]
        I3 = sigma[0, 0] * sigma[1, 1] * sigma[2, 2]

        # 定义不变量对于F的导数
        R = U @ V.transpose()
        col0 = tm.vec3(F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1],
                       F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2],
                       F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0])
        col1 = tm.vec3(F[2, 1] * F[0, 2] - F[2, 2] * F[0, 1],
                       F[2, 2] * F[0, 0] - F[2, 0] * F[0, 2],
                       F[2, 0] * F[0, 1] - F[2, 1] * F[0, 0])
        col2 = tm.vec3(F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1],
                       F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2],
                       F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0])
        dI1dF = R
        dI2dF = 2 * F
        dI3dF = tm.mat3([col0, col1, col2])

        # 定义能量密度
        # ψ = μ / 2 * (I2 - 3) - μ(I3 - 1) + λ / 2 * (I3 - 1) ^ 2
        Psi = mu / 2. * (I2 - 3.) - mu * (I3 - 1.) + la / 2. * (I3 - 1.) * (I3 - 1.)

        # 定义1st Piola-Kirchhoff tensor
        # P = μ / 2 * dI2dF - μ * dI3dF + λ * (I3 - 1) * dI3dF
        P = mu / 2. * dI2dF - mu * dI3dF + la * (I3 - 1.) * dI3dF

        return Psi, P


@ti.data_oriented
class Stable_Neo_Hookean_with_active:
    """
    elastic energy density:
    I1 = tr(S), I2 = tr(F^T@F), I3 = det(F)
    ψ = μ/2 * (I2-3) - μ(I3-1) + λ/2 * (I3-1)^2
    """

    def __init__(self, Youngs_modulus, Poisson_ratio, active_tension):
        self.Youngs_modulus = Youngs_modulus
        self.Poisson_ratio = Poisson_ratio
        self.LameLa = Youngs_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
        self.LameMu = Youngs_modulus / (2 * (1 + Poisson_ratio))
        self.Ta = active_tension

    @ti.func
    def ComputePsiDeriv(self, deformation_gradient: ti.template(), fiber_direction: ti.template()):
        """
        input deformationGradient F,
        return Energy density Psi and the first Piola-Kirchhoff tensor P
        """
        mu, la = ti.static(self.LameMu, self.LameLa)

        F = deformation_gradient
        f0 = fiber_direction
        # J = F.determinant()

        # 修改反转元素
        U, sigma, V = ti.svd(F, ti.f32)
        if sigma[2, 2] < 0:
            sigma[2, 2] = -sigma[2, 2]

        # 定义不变量: I1 = tr(S), I2 = tr(F^T@F), I3 = det(F)
        I1 = sigma[0, 0] + sigma[1, 1] + sigma[2, 2]
        I2 = sigma[0, 0] * sigma[0, 0] + sigma[1, 1] * sigma[1, 1] + sigma[2, 2] * sigma[2, 2]
        I3 = sigma[0, 0] * sigma[1, 1] * sigma[2, 2]

        # 定义不变量对于F的导数
        R = U @ V.transpose()
        col0 = tm.vec3(F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1],
                       F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2],
                       F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0])
        col1 = tm.vec3(F[2, 1] * F[0, 2] - F[2, 2] * F[0, 1],
                       F[2, 2] * F[0, 0] - F[2, 0] * F[0, 2],
                       F[2, 0] * F[0, 1] - F[2, 1] * F[0, 0])
        col2 = tm.vec3(F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1],
                       F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2],
                       F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0])
        dI1dF = R
        dI2dF = 2 * F
        dI3dF = tm.mat3([col0, col1, col2])

        # 定义能量密度
        # ψ = μ / 2 * (I2 - 3) - μ(I3 - 1) + λ / 2 * (I3 - 1) ^ 2
        Psi = mu / 2. * (I2 - 3.) - mu * (I3 - 1.) + la / 2. * (I3 - 1.) * (I3 - 1.)

        # 定义1st Piola-Kirchhoff tensor
        # P_pass = μ / 2 * dI2dF - μ * dI3dF + λ * (I3 - 1) * dI3dF
        P_pass = mu / 2. * dI2dF - mu * dI3dF + la * (I3 - 1.) * dI3dF
        # P_act = Ta * (F@f0)@(f0^T) / sqrt(I4f)
        f = (F @ f0)
        I4f = f[0] * f[0] + f[1] * f[1] + f[2] * f[2]
        P_act = self.Ta * (F @ f0) @ (f0.transpose()) / tm.sqrt(I4f)
        P = P_pass + P_act

        return Psi, P


@ti.data_oriented
class Holzapfel_Odgen_2009:
    """
    elastic energy density:
    I1 = tr(C), I2 = 0.5 * (I1^2 - tr(C^2)), I3 = det(C) = J^2
    Iff = C : f0 otimes f0 = f0^T @ F^T @ F @ f0, Iss = C : s0 otimes s0, Ifs = C : f0 otimes s0
    P = P_passive + P_active
    Psi_passive = a / (2 * b) * exp(b * (I1 - 3)) - a * ln(J) + 0.5 * lambda * (ln(J))^2 +
                  sum_{i=f,s}{0.5 * a_i / b_i * (exp(b_i * (Iii - 1)^2) - 1)} +
                  0.5 * a_fs / b_fs * (exp(b_fs * Ifs^2) - 1)
    P_passive = F @ S_passive
    S_passive = a * exp(b * (I1 - 3)) * I + {lambda * ln(J) - a} * C^-1 +
                2 * a_f * (Iff - 1) * exp(b_f * (Iff - 1)^2) * f0 otimes f0 +
                2 * a_s * (Iss - 1) * exp(b_s * (Iss - 1)^2) * s0 otimes s0 +
                a_fs * Ifs * exp(b_fs * Ifs^2) * (f0 otimes s0 + s0 otimes f0)
    P_active = Ta * F * f0 otimes f0
    Psi_active = Ta / 2 * (Iff - 1)
    Ta' = epsilon(Vm) * (ka * (Vm - Vr) - Ta)
    epsilon(Vm) = epsilon_0 for Vm < 0.05 || 10 * epsilon_0 for Vm >= 0.05
    epsilon_0 = 1
    Ka = 47.9 kPa
    V = 100 * Vm - 80
    """

    def __init__(self, a, b, a_f, b_f, a_s, b_s, a_fs, b_fs, LameLa):
        self.a = a
        self.b = b
        self.a_f = a_f
        self.b_f = b_f
        self.a_s = a_s
        self.b_s = b_s
        self.a_fs = a_fs
        self.b_fs = b_fs
        self.LameLa = LameLa

    @ti.func
    def ComputePsi(self, deformation_gradient: ti.template(), fiber_direction: ti.template(),
                   sheet_direction: ti.template(), Ta: ti.template()):
        """
        input deformationGradient F,
        return Energy density Psi
        """
        a, b, a_f, b_f = ti.static(self.a, self.b, self.a_f, self.b_f)
        a_s, b_s, a_fs, b_fs = ti.static(self.a_s, self.b_s, self.a_fs, self.b_fs)
        LameLa = ti.static(self.LameLa)

        F = deformation_gradient
        # 修改反转元素
        U, sigma, V = ti.svd(F, ti.f32)
        if sigma[2, 2] < 0:
            sigma[2, 2] = -sigma[2, 2]
        F = U @ sigma @ V.transpose()

        f0 = fiber_direction
        s0 = sheet_direction
        f = F @ f0
        s = F @ s0
        J = F.determinant()
        C = F.transpose() @ F

        # 定义不变量:
        # I1 = tr(C)
        I1 = C[0, 0] + C[1, 1] + C[2, 2]
        # Iff = C: f0 otimes f0, Iss = C: s0 otimes s0, Ifs = C: f0 otimes s0
        Iff = f.dot(f)
        Iss = s.dot(s)
        Ifs = s.dot(f)

        # 定义能量密度
        # Psi_passive = a / (2 * b) * exp(b * (I1 - 3)) - a * ln(J) + 0.5 * lambda * (ln(J)) ^ 2 +
        #               sum_{i=f, s}{0.5 * a_i / b_i * (exp(b_i * (Iii - 1) ^ 2) - 1)} + \
        #               0.5 * a_fs / b_fs * (exp(b_fs * Ifs ^ 2) - 1)
        Psi_passive = a * 0.5 / b * ti.exp(b * (I1 - 3)) - a * ti.log(J) + 0.5 * LameLa * (ti.log(J) ** 2) + \
                      0.5 * a_f / b_f * (ti.exp(b_f * (Iff - 1.0) ** 2) - 1) + \
                      0.5 * a_s / b_s * (ti.exp(b_s * (Iss - 1.0) ** 2) - 1) + \
                      0.5 * a_fs / b_fs * (ti.exp(b_fs * (Ifs ** 2)) - 1)
        # Psi_active = Ta / 2 * (Iff - 1)
        Psi_active = Ta * 0.5 * (Iff - 1.0)
        Psi = Psi_passive + Psi_active

        return Psi

    @ti.func
    def ComputePsiDeriv(self, deformation_gradient: ti.template(), fiber_direction: ti.template(),
                        sheet_direction: ti.template(), Ta: ti.template()):
        """
        input deformationGradient F,
        return Energy density Psi and the first Piola-Kirchhoff tensor P
        """
        a, b, a_f, b_f = ti.static(self.a, self.b, self.a_f, self.b_f)
        a_s, b_s, a_fs, b_fs = ti.static(self.a_s, self.b_s, self.a_fs, self.b_fs)
        LameLa = ti.static(self.LameLa)

        F = deformation_gradient
        # 修改反转元素
        U, sigma, V = ti.svd(F, ti.f32)
        if sigma[2, 2] < 0:
            sigma[2, 2] = -sigma[2, 2]
        F = U @ sigma @ V.transpose()

        f0 = fiber_direction
        s0 = sheet_direction
        f = F @ f0
        s = F @ s0
        J = F.determinant()
        C = F.transpose() @ F
        C_inv = C.inverse()
        C2 = C @ C

        # 定义不变量:
        # I1 = tr(C)
        I1 = C[0, 0] + C[1, 1] + C[2, 2]
        # Iff = C: f0 otimes f0, Iss = C: s0 otimes s0, Ifs = C: f0 otimes s0
        Iff = f.dot(f)
        Iss = s.dot(s)
        Ifs = s.dot(f)

        # 定义能量密度
        # Psi_passive = a / (2 * b) * exp(b * (I1 - 3)) - a * ln(J) + 0.5 * lambda * (ln(J)) ^ 2 +
        #               sum_{i=f, s}{0.5 * a_i / b_i * (exp(b_i * (Iii - 1) ^ 2) - 1)} + \
        #               0.5 * a_fs / b_fs * (exp(b_fs * Ifs ^ 2) - 1)
        Psi_passive = a * 0.5 / b * ti.exp(b * (I1 - 3)) - a * ti.log(J) + 0.5 * LameLa * (ti.log(J) ** 2) + \
                                0.5 * a_f / b_f * (ti.exp(b_f * (Iff - 1.0)**2) - 1) + \
                                0.5 * a_s / b_s * (ti.exp(b_s * (Iss - 1.0)**2) - 1) + \
                                0.5 * a_fs / b_fs * (ti.exp(b_fs * (Ifs**2)) - 1)
        # Psi_active = Ta / 2 * (Iff - 1)
        Psi_active = Ta * 0.5 * (Iff - 1.0)
        Psi = Psi_passive + Psi_active

        # 定义1st Piola-Kirchhoff tensor
        # P_passive = F @ S_passive
        # S_passive = a * exp(b * (I1 - 3)) * I + {lambda *ln(J) - a} * C ^ -1 +
        #             2 * a_f * (Iff - 1) * exp(b_f * (Iff - 1) ^ 2) * f0 otimes f0 + \
        #             2 * a_s * (Iss - 1) * exp(b_s * (Iss - 1) ^ 2) * s0 otimes s0 +
        #             a_fs * Ifs * exp(b_fs * Ifs ^ 2) * (f0 otimes s0 + s0 otimes f0)
        Identity3 = tm.mat3([1., 0., 0.], [0., 1., 0.], [0., 0., 1.])

        S_passive = a * ti.exp(b * (I1 - 3.0)) * Identity3 + (LameLa * tm.log(J) - a) * C_inv + \
                        2. * a_f * (Iff - 1.) * tm.exp(b_f * ((Iff - 1.) ** 2)) * (f0.outer_product(f0)) + \
                        2. * a_s * (Iss - 1.) * tm.exp(b_s * ((Iss - 1.) ** 2)) * (s0.outer_product(s0)) + \
                        a_fs * Ifs * tm.exp(b_fs * (Ifs ** 2)) * (f0.outer_product(s0) + s0.outer_product(f0))
        P_passive = F @ S_passive

        # P_active = Ta * F * f0 otimes f0
        P_active = Ta * F @ (f0.outer_product(f0))
        P = P_passive + P_active

        return Psi, P


@ti.data_oriented
class Stable_Neo_Hookean_with_active_stress:
    """
    elastic energy density:
    I1 = tr(S), I2 = tr(F^T@F), I3 = det(F)
    ψ = μ/2 * (I2-3) - μ(I3-1) + λ/2 * (I3-1)^2
    """

    def __init__(self, Youngs_modulus, Poisson_ratio):
        self.Youngs_modulus = Youngs_modulus
        self.Poisson_ratio = Poisson_ratio
        self.LameLa = Youngs_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
        self.LameMu = Youngs_modulus / (2 * (1 + Poisson_ratio))

    @ti.func
    def ComputePsi(self, deformation_gradient: ti.template(), fiber_direction: ti.template(),
                        sheet_direction: ti.template(), Ta: ti.template()):
        """
        input deformationGradient F,
        return Energy density Psi
        """
        mu, la = ti.static(self.LameMu, self.LameLa)

        F = deformation_gradient
        f0 = fiber_direction
        # J = F.determinant()

        # 修改反转元素
        U, sigma, V = ti.svd(F, ti.f32)
        if sigma[2, 2] < 0:
            sigma[2, 2] = -sigma[2, 2]

        # 定义不变量: I1 = tr(S), I2 = tr(F^T@F), I3 = det(F)
        I1 = sigma[0, 0] + sigma[1, 1] + sigma[2, 2]
        I2 = sigma[0, 0] * sigma[0, 0] + sigma[1, 1] * sigma[1, 1] + sigma[2, 2] * sigma[2, 2]
        I3 = sigma[0, 0] * sigma[1, 1] * sigma[2, 2]

        # 定义不变量对于F的导数
        R = U @ V.transpose()
        col0 = tm.vec3(F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1],
                       F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2],
                       F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0])
        col1 = tm.vec3(F[2, 1] * F[0, 2] - F[2, 2] * F[0, 1],
                       F[2, 2] * F[0, 0] - F[2, 0] * F[0, 2],
                       F[2, 0] * F[0, 1] - F[2, 1] * F[0, 0])
        col2 = tm.vec3(F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1],
                       F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2],
                       F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0])
        dI1dF = R
        dI2dF = 2 * F
        dI3dF = tm.mat3([col0, col1, col2])

        # 定义能量密度
        # ψ = μ / 2 * (I2 - 3) - μ(I3 - 1) + λ / 2 * (I3 - 1) ^ 2
        Psi_passive = mu / 2. * (I2 - 3.) - mu * (I3 - 1.) + la / 2. * (I3 - 1.) * (I3 - 1.)
        f = F @ f0
        Iff = f.dot(f)
        Psi_active = Ta * 0.5 * (Iff - 1.0)
        Psi = Psi_passive + Psi_active

        return Psi

    @ti.func
    def ComputePsiDeriv(self, deformation_gradient: ti.template(), fiber_direction: ti.template(),
                        sheet_direction: ti.template(), Ta: ti.template()):
        """
        input deformationGradient F,
        return Energy density Psi and the first Piola-Kirchhoff tensor P
        """
        mu, la = ti.static(self.LameMu, self.LameLa)

        F = deformation_gradient
        f0 = fiber_direction
        # J = F.determinant()

        # 修改反转元素
        U, sigma, V = ti.svd(F, ti.f32)
        if sigma[2, 2] < 0:
            sigma[2, 2] = -sigma[2, 2]

        # 定义不变量: I1 = tr(S), I2 = tr(F^T@F), I3 = det(F)
        I1 = sigma[0, 0] + sigma[1, 1] + sigma[2, 2]
        I2 = sigma[0, 0] * sigma[0, 0] + sigma[1, 1] * sigma[1, 1] + sigma[2, 2] * sigma[2, 2]
        I3 = sigma[0, 0] * sigma[1, 1] * sigma[2, 2]

        # 定义不变量对于F的导数
        R = U @ V.transpose()
        col0 = tm.vec3(F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1],
                       F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2],
                       F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0])
        col1 = tm.vec3(F[2, 1] * F[0, 2] - F[2, 2] * F[0, 1],
                       F[2, 2] * F[0, 0] - F[2, 0] * F[0, 2],
                       F[2, 0] * F[0, 1] - F[2, 1] * F[0, 0])
        col2 = tm.vec3(F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1],
                       F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2],
                       F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0])
        dI1dF = R
        dI2dF = 2 * F
        dI3dF = tm.mat3([col0, col1, col2])

        # 定义能量密度
        # ψ = μ / 2 * (I2 - 3) - μ(I3 - 1) + λ / 2 * (I3 - 1) ^ 2
        Psi_passive = mu / 2. * (I2 - 3.) - mu * (I3 - 1.) + la / 2. * (I3 - 1.) * (I3 - 1.)
        f = F @ f0
        Iff = f.dot(f)
        Psi_active = Ta * 0.5 * (Iff - 1.0)
        Psi = Psi_passive + Psi_active

        # 定义1st Piola-Kirchhoff tensor
        # P_passive = μ / 2 * dI2dF - μ * dI3dF + λ * (I3 - 1) * dI3dF
        P_passive = mu / 2. * dI2dF - mu * dI3dF + la * (I3 - 1.) * dI3dF
        # P_act = Ta * (F@f0)@(f0^T) / sqrt(I4f)
        f = (F @ f0)
        I4f = f[0] * f[0] + f[1] * f[1] + f[2] * f[2]
        # P_active = self.Ta * (F @ f0) @ (f0.transpose()) / tm.sqrt(I4f)
        P_active = Ta * F @ (f0.outer_product(f0))
        P = P_passive + P_active

        return Psi, P


@ti.kernel
def debug(material: ti.template()):
    F = tm.mat3([1, 0, 0,
                 0, 1, 0,
                 0, 0, 1])
    f0 = tm.vec3([1, 0.5, 0.2])
    s0 = tm.vec3([0, 1, 0])
    Psi, P = material.ComputePsiDeriv(F, f0, s0, 60)
    print(Psi, P)


if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f32, kernel_profiler=True)
    # Youngs_Modulus = 1000.
    # Poisson_Ratio = 0.49
    # # material = Stable_Neo_Hookean(Youngs_modulus=Youngs_Modulus, Poisson_ratio=Poisson_Ratio)
    # material = Stable_Neo_Hookean_with_active(Youngs_modulus=Youngs_Modulus, Poisson_ratio=Poisson_Ratio,
    #                                           active_tension=60)
    # debug(material)

    material = Holzapfel_Odgen_2009(a=0.059, b=8.023, a_f=18.472, b_f=16.026, a_s=2.841, b_s=11.12,
                                                        a_fs=0.216, b_fs=11.436, LameLa=10000)

    debug(material)
