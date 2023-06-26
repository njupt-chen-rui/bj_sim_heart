import taichi as ti
import taichi.math as tm


@ti.data_oriented
class A:
    def __init__(self):
        self.a = ti.field(float, shape=(3,))
        self.init()

    @ti.kernel
    def init(self):
        for i in self.a:
            self.a[i] = 1.0

    @ti.kernel
    def update(self):
        for i in self.a:
            self.a[i] = 3.0


@ti.data_oriented
class B:
    def __init__(self, base: A):
        self.a = base.a
        self.init()

    @ti.kernel
    def init(self):
        for i in self.a:
            self.a[i] = 2.0


if __name__ == "__main__":
    ti.init(arch=ti.cuda)
    # pos = ti.Vector.field(3, float, shape=(4,))
    # # Ds = tm.mat3(pos[1]-pos[0], pos[2]-pos[0], pos[3]-pos[0])
    # pos[0] = tm.vec3(1., 0., 0.)
    # pos[1] = tm.vec3(0., 1., 0.)
    # print(pos[1].outer_product(pos[0]))
    # Youngs_modulus = 10000.0
    # Poisson_ratio = 0.49
    # LameLa = Youngs_modulus * Poisson_ratio / ((1 + Poisson_ratio) * (1 - 2 * Poisson_ratio))
    # LameMu = Youngs_modulus / (2 * (1 + Poisson_ratio))
    # print(LameLa, LameMu)
    a = A()
    b = B(a)
    a.update()
    print(b.a)
