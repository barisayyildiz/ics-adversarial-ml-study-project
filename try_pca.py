import task1

A = [1, -1, 2, -2]
B = [1, -1, 1, -1]

mean, cov = task1.covariance({"A": A, "B": B})
print("covariance matrix:", cov)

values, vectors = task1.jacobi_eigen(cov)
print("eigenvalues:", values)
print("weights of new1:", vectors[0])
print("weights of new2:", vectors[1])

new1 = [vectors[0][0] * a + vectors[0][1] * b for a, b in zip(A, B)]
new2 = [vectors[1][0] * a + vectors[1][1] * b for a, b in zip(A, B)]
print("new1:", new1)
print("new2:", new2)

_, cov_new = task1.covariance({"new1": new1, "new2": new2})
print("covariance matrix of the new columns:", cov_new)
