package net.librec.recommender;


import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;

import com.google.common.collect.*;
import net.librec.annotation.ModelData;
import net.librec.common.LibrecException;
import net.librec.math.structure.*;


@ModelData({"isRanking", "trainMatrix"})
public class NaiveBayesRecommender extends AbstractRecommender {

    private static final int BSIZE = 1024 * 1024;

    protected SparseMatrix m_featureMatrix;
    protected double m_threshold;
    protected DenseMatrix probFeatureLike;
    protected DenseMatrix probFeatureDislike;
    protected double[] probLikeMatrix;
    protected double[] probDislikeMatrix;

    @Override
    public void setup() throws LibrecException {
        // out of mercy this method was given
        super.setup();

        // we use this to set the threshold of what is a liked
        m_threshold = conf.getDouble("rec.rating.threshold");

        String contentPath = conf.get("dfs.content.path");
        Table<Integer, Integer, Integer> contentTable = HashBasedTable.create();
        HashBiMap<String, Integer> itemIds = HashBiMap.create();
        HashBiMap<String, Integer> featureIds = HashBiMap.create();

        try {

            FileInputStream fileInputStream = new FileInputStream(contentPath);
            FileChannel fileRead = fileInputStream.getChannel();
            ByteBuffer buffer = ByteBuffer.allocate(BSIZE);
            int len;
            String bufferLine = new String();
            byte[] bytes = new byte[BSIZE];
            while ((len = fileRead.read(buffer)) != -1) {
                buffer.flip();
                buffer.get(bytes, 0, len);
                bufferLine = bufferLine.concat(new String(bytes, 0, len));
                String[] bufferData = bufferLine.split(System.getProperty("line.separator") + "+");
                boolean isComplete = bufferLine.endsWith(System.getProperty("line.separator"));
                int loopLength = isComplete ? bufferData.length : bufferData.length - 1;
                for (int i = 0; i < loopLength; i++) {
                    String line = new String(bufferData[i]);
                    String[] data = line.trim().split("[ \t,]+");

                    String item = data[0];
                    // inner id starting from 0
                    int row = itemIds.containsKey(item) ? itemIds.get(item) : itemIds.size();
                    itemIds.put(item, row);

                    for (int j = 1; j < data.length; j++) {
                        String feature = data[j];
                        int col = featureIds.containsKey(feature) ? featureIds.get(feature) : featureIds.size();
                        featureIds.put(feature, col);

                        contentTable.put(row, col, 1);
                    }

                }
            }
        } catch (IOException e) {
            LOG.error("Error reading file: " + contentPath + e);
            throw (new LibrecException(e));
        }

        m_featureMatrix = new SparseMatrix(itemIds.size(), featureIds.size(), contentTable);
        LOG.info("Loaded item features from " + contentPath);
    }

    public void trainModel() throws LibrecException {

        // number of users and features
        int numUsers = trainMatrix.numRows();
        int numFeatures = m_featureMatrix.numColumns();

        // store the prob_like
        // store the prob_dislike
        probLikeMatrix = new double[numUsers];
        probDislikeMatrix = new double[numUsers];

        // store the probability of feature given like
        // store the probability of feature given dislike
        probFeatureLike = new DenseMatrix(numUsers, numFeatures);
        probFeatureDislike = new DenseMatrix(numUsers, numFeatures);

        // these are used to say if the user liked
        // or disliked the items
        int itemsLikes;
        int itemsDislike;

        for (int u = 0; u < numUsers; u++) {

            // initialize like and dislike counter to 0
            itemsLikes = 0;
            itemsDislike = 0;

            // likes and dislikes for each feature
            int[] featuresLikes = new int[numFeatures];
            int[] featuresDislike = new int[numFeatures];

            // initialize the hashmaps to 0
            Arrays.fill(featuresLikes, new Integer(0));
            Arrays.fill(featuresDislike, new Integer(0));

            // gather the list of features for each item
            SparseVector itemVector = trainMatrix.row(u);
            Iterator<VectorEntry> item_ir = itemVector.iterator();

            while (item_ir.hasNext()) {

                VectorEntry ratingEntry = item_ir.next();
                double rating = ratingEntry.get();
                int item = ratingEntry.index();

                // see if the item was liked or disliked and
                // increment accordingly
                if (rating > m_threshold) {
                    itemsLikes++;
                } else if (rating > 0) {
                    itemsDislike++;
                }

                // get all the features for the item
                SparseVector featureVector = m_featureMatrix.row(item);
                Iterator<VectorEntry> feature_ir = featureVector.iterator();

                // go through all the features and increment the like
                // counters for the feature or dislike
                while (feature_ir.hasNext()) {

                    VectorEntry featureEntry = feature_ir.next();
                    int feature = featureEntry.index();

                    if (rating > m_threshold) {
                        featuresLikes[feature]++;
                    } else if (rating > 0) {
                        featuresDislike[feature]++;
                    }
                }

                // calculate the probability of like and dislike for the user
                double total_ratings = itemsLikes + itemsDislike;
                double prob_like = (double) (itemsLikes + 1) / (total_ratings + 2);
                double prob_dislike = (double) (itemsDislike + 1) / (total_ratings + 2);

                // add to the probability matrices
                probLikeMatrix[u] = prob_like;
                probDislikeMatrix[u] = prob_dislike;


                for (int f = 0; f < numFeatures; f++) {

                    // get the probabilites of the feature given like and dislike
                    double prob_feature_like = (double) (featuresLikes[f] + 1) / (itemsLikes + 1);
                    double prob_feature_dislike = (double) (featuresDislike[f] + 1) / (itemsDislike + 1);

                    // set the values in the matrix
                    probFeatureLike.set(u, f, prob_feature_like);
                    probFeatureDislike.set(u, f, prob_feature_dislike);

                }
            }
        }
    }

    @Override
    public double predict(int user, int item) throws LibrecException {

        // running prob like * prob feature given like
        // running prob dislike * prob feature given dislike
        double run_prob_dislike = 0;
        double run_prob_like = 0;

        // get the user probability of like and dislike
        double prob_like = probLikeMatrix[user];
        double prob_dislike = probDislikeMatrix[user];

        // get the features for the user
        SparseVector featureVector = m_featureMatrix.row(item);
        List<Integer> featureIndexVect = featureVector.getIndexList();

        for (int feature : featureVector.getIndexList()) {

            // used the calculate the k constant

            // probability of feature given like
            // probability of feature given dislike
            double like_feature = probFeatureLike.get(user, feature);
            double dislike_feature = probFeatureDislike.get(user, feature);

            // intermediate like and dislike values
            double inter_like = like_feature * prob_like;
            double inter_dislike = dislike_feature * prob_dislike;

            // increment totals
            run_prob_like += inter_like;
            run_prob_dislike += inter_dislike;

        }

        // calculate k
        double k = 1/(run_prob_like + run_prob_dislike);

        // calculate probability of like given feature
        // calculate probability of dislike given feature
        double prob_like_feature = k * run_prob_like;
        double prob_dislike_feature = k * run_prob_dislike;

        // get the log likliehood
        double logliklihood = Math.log(prob_like_feature / prob_dislike_feature);

        // use logit to get a probability
        double prob_like_final = Math.exp(logliklihood) / (1 + Math.exp(logliklihood));

        // make the final prediction
        return minRate + prob_like_final * (maxRate - minRate);

    }
}

