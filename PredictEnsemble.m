function [Yhat] = PredictEnsemble(Yhatlearners,beta,my)

Yhat = Yhatlearners * beta + my;

end