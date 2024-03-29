run('/Users/sourabhdaptardar/Projects/vlfeat/toolbox/vl_setup')
numFeatures = 5000 ;
dimension = 2 ;
data = rand(dimension,numFeatures) ;

numClusters = 30 ;
[means, covariances, priors] = vl_gmm(data, numClusters);

numDataToBeEncoded = 1000;
dataToBeEncoded = rand(dimension,numDataToBeEncoded);

encoding = vl_fisher(dataToBeEncoded, means, covariances, priors);
