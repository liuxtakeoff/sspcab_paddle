from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity
import pickle
import paddle


class Density(object):
    def fit(self, embeddings):
        raise NotImplementedError

    def predict(self, embeddings):
        raise NotImplementedError


class GaussianDensityPaddle(object):
    """Gaussian Density estimation similar to the implementation used by Ripple et al.
    The code of Ripple et al. can be found here: https://github.com/ORippler/gaussian-ad-mvtec.
    """
    def load(self,param_path):
        with open(param_path,"rb") as f:
            self.params = pickle.load(f)
            self.min = 0
            self.max = 1
    def fit(self, embeddings,param_savepath):
        self.mean = paddle.mean(embeddings, axis=0)
        self.inv_cov = paddle.to_tensor(LedoitWolf().fit(embeddings.cpu()).precision_, place="cpu",dtype="float32")
        self.params = {"mean": self.mean.cpu().detach().numpy(), "inv_cov": self.inv_cov.cpu().detach().numpy()}
        with open(param_savepath,"wb") as f:
            pickle.dump(self.params,f)
        self.min = None
        self.max = None

    def predict(self, embeddings):
        distances = self.mahalanobis_distance(embeddings, paddle.to_tensor(self.params["mean"]), paddle.to_tensor(self.params["inv_cov"]))
        if self.min  == None:
            return distances
        else:
            distances = (distances - self.min) / (self.max - self.min + 1e-8)
            return distances
    @staticmethod
    def mahalanobis_distance(
            values: paddle.Tensor, mean: paddle.Tensor, inv_covariance: paddle.Tensor
    ) -> paddle.Tensor:
        """Compute the batched mahalanobis distance.
        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.

        from https://github.com/ORippler/gaussian-ad-mvtec/blob/4e85fb5224eee13e8643b684c8ef15ab7d5d016e/src/gaussian/model.py#L308
        """
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert len(inv_covariance.shape) == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        dist = paddle.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
        return dist.sqrt()


class GaussianDensitySklearn():
    """Li et al. use sklearn for density estimation.
    This implementation uses sklearn KernelDensity module for fitting and predicting.
    """
    def load(self,model_path):
        with open(model_path,"rb") as f:
            self.kde = pickle.load(f)
            self.min = 470
            self.max = 471
    def fit(self, embeddings,save_path):
        # estimate KDE parameters
        # use grid search cross-validation to optimize the bandwidth
        self.min = None
        self.max = None
        self.kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(embeddings)
        with open(save_path,"wb") as f:
            pickle.dump(self.kde,f)
    def predict(self, embeddings):
        scores = self.kde.score_samples(embeddings)

        # invert scores, so they fit to the class labels for the auc calculation
        scores = -scores
        if self.min == None:
            return scores
        else:
            scores = (scores-self.min)/(self.max-self.min+1e-8)
            return scores