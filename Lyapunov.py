import warnings
from nolitsa import data, lyapunov
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition


def poly_fit(x, y, degree, fit="RANSAC"):
    # check if we can use RANSAC
    if fit == "RANSAC":
        try:
            # ignore ImportWarnings in sklearn
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ImportWarning)
                import sklearn.linear_model as sklin
                import sklearn.preprocessing as skpre
        except ImportError:
            warnings.warn(
                "fitting mode 'RANSAC' requires the package sklearn, using"
                + " 'poly' instead",
                RuntimeWarning)
            fit = "poly"

    if fit == "poly":
        return np.polyfit(x, y, degree)
    elif fit == "RANSAC":
        model = sklin.RANSACRegressor(sklin.LinearRegression(fit_intercept=False))
        xdat = np.asarray(x)
        if len(xdat.shape) == 1:
            # interpret 1d-array as list of len(x) samples instead of
            # one sample of length len(x)
            xdat = xdat.reshape(-1, 1)
        polydat = skpre.PolynomialFeatures(degree).fit_transform(xdat)
        try:
            model.fit(polydat, y)
            coef = model.estimator_.coef_[::-1]
        except ValueError:
            warnings.warn(
                "RANSAC did not reach consensus, "
                + "using numpy's polyfit",
                RuntimeWarning)
            coef = np.polyfit(x, y, degree)
        return coef
    else:
        raise ValueError("invalid fitting mode ({})".format(fit))


dt = 1/32
x0 = [0.62225717, -0.08232857, 30.60845379]
x = data.lorenz(length=4000, sample=dt, x0=x0,
                sigma=16.0, beta=4.0, rho=45.92)[1]
# plt.plot(range(len(x)), x)
# plt.show()

# Choose appropriate Theiler window.
meanperiod = 50
maxt = 500

mm = MinMaxScaler(feature_range=(-1, 1))
for i in range(18,50):
    if i<10:
        data = np.loadtxt('clean_bvp_all/s0' + str(i) + '/clean_bvp_s' + str(i) + '_T1.csv', delimiter=',')
    else:
        data=np.loadtxt('clean_bvp_all/s'+str(i)+'/clean_bvp_s'+str(i)+'_T1.csv',delimiter=',')
    data=data.reshape(len(data),1)
    # data=mm.fit_transform(data)
    # bvp_data = data.reshape((len(data),))
    # dim = 7
    # tau = 8
    # ps_vector = np.zeros(((len(data) - (dim - 1) * tau), dim))
    # for j in range(len(data) - (dim - 1) * tau):
    #     for k in range(dim):
    #         ps_vector[j][k] = bvp_data[j + k * tau]
    # result = ps_vector
    # model = decomposition.PCA(n_components=3)
    # model.fit(result)
    # result = model.fit_transform(result)
    # result = mm.fit_transform(result)

    d = lyapunov.mle(data, maxt=maxt, window=meanperiod)
    t = np.arange(maxt) * dt
    coefs = poly_fit(t, d, 1)

    print('LLE = ', coefs[0])

# data =np.load('datasets/bvp_data/bvp_all_1.npy', allow_pickle=True)
# d = lyapunov.mle(data, maxt=maxt, window=meanperiod)
# t = np.arange(maxt) * dt
# coefs = poly_fit(t, d, 1)
# print('LLE = ', coefs[0])


# plt.title('Maximum Lyapunov exponent for the Lorenz system')
# plt.xlabel(r'Time $t$')
# plt.ylabel(r'Average divergence $\langle d_i(t) \rangle$')
# plt.plot(t, d, label='divergence')
# plt.plot(t, t * 1.50, '--', label='slope=1.5')
# plt.plot(t, coefs[1] + coefs[0] * t, '--', label='RANSAC')
# plt.legend()
# plt.show()
