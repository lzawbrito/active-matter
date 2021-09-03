from autograd import grad 

gradf = grad(lambda x: x[0] ** 2 + x[1] ** 2)

sol = gradf((1.0, 2.0))

print(sol[0])







    
    