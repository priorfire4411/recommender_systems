/**
 * Copyright (C) 2016 LibRec
 * <p>
 * This file is part of LibRec.
 * LibRec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * LibRec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with LibRec. If not, see <http://www.gnu.org/licenses/>.
 *
 * edited by Brian Craft in the winter of 2017 to add in signifigance
 * weighting
 */
package net.librec.recommender.cf;

import net.librec.annotation.ModelData;
import net.librec.common.LibrecException;
import net.librec.math.structure.DenseVector;
import net.librec.math.structure.SparseVector;
import net.librec.math.structure.SymmMatrix;
import net.librec.math.structure.VectorEntry;
import net.librec.recommender.AbstractRecommender;
import net.librec.util.Lists;

import java.awt.*;
import java.util.*;
import java.util.List;
import java.util.Map.Entry;

/**
 * UserKNNRecommender
 *
 * @author WangYuFeng and Keqiang Wang
 */
@ModelData({"isRanking", "knn", "userMappingData", "itemMappingData", "userMeans", "trainMatrix", "similarityMatrix"})
public class UserKNNRecommender extends AbstractRecommender {

    // this will be used to store the signifigance weighting coefficients for
    // each user.  in this case, there is 1508 users, so each
    // key will be an array list of 1508 values
    private HashMap overLapMatrix = new HashMap<Integer, ArrayList>();
    private int knn;
    private DenseVector userMeans;
    private SymmMatrix similarityMatrix;
    private List<Map.Entry<Integer, Double>>[] userSimilarityList;

    /**
     * (non-Javadoc)
     *
     * @see net.librec.recommender.AbstractRecommender#setup()
     */
    @Override
    protected void setup() throws LibrecException {
        super.setup();
        knn = conf.getInt("rec.neighbors.knn.number");
        similarityMatrix = context.getSimilarity().getSimilarityMatrix();
    }

    /**
     * (non-Javadoc)
     *
     * @see net.librec.recommender.AbstractRecommender#trainModel()
     */
    @Override
    protected void trainModel() throws LibrecException {

        // overlap hash matrix will loop through each users rated items
        // and proceed to find the overlap with each user
        // the results will be stored in a hashmap where the key
        // is a user and the value is an array list in which a
        // which the 0 index is the overlap with the first user
        // we just leverage the loop that is already creating
        // the usermeans for each row
        double overLapValue; // store the number of overlapping ratings
        double coef; // store the significance weighting coefficient
        double beta; // store the beta value
        List<Integer> vect1 = new ArrayList<Integer>(); // store the indexed items for user 1
        List<Integer> vect2 = new ArrayList<Integer>(); // store the indexed items for user 2
        beta = conf.getInt("beta");

        userMeans = new DenseVector(numUsers);

        for (int userIdx = 0; userIdx < numUsers; userIdx++) {

            // this will store the overlap values for each user
            List<Double> vect3 = new ArrayList<Double>();
            SparseVector userRatingVector = trainMatrix.row(userIdx);
            userMeans.set(userIdx, userRatingVector.getCount() > 0 ? userRatingVector.mean() : globalMean);
            vect1 = trainMatrix.row(userIdx).getIndexList();

            for (int i = 0; i < numUsers; i++){
                    overLapValue = 0;
                    vect2 = trainMatrix.row(i).getIndexList();

                    for ( int t : vect1){

                        if(vect2.contains(t)){
                            overLapValue++;
                        }else{
                            continue;
                        }
                    }

                    if(overLapValue >= beta) {
                        coef = 1;
                        vect3.add(coef);
                    }else{
                        coef = overLapValue/beta;
                        vect3.add(coef);
                    }
                }

            overLapMatrix.put(userIdx,vect3);

            // here we begin looping through the similarity matrix
            // and finding the coefficient values to multiple the similarities
            // by then we replace the values of the matrix with the new value
            for (int r = 0; r < numUsers; r++){
                    double new_sim;
                    new_sim = similarityMatrix.get(userIdx,r) * vect3.get(r);
                    similarityMatrix.set(userIdx,r,new_sim);
                }
        }
    }

    /**
     * (non-Javadoc)
     *
     * @see net.librec.recommender.AbstractRecommender#predict(int, int)
     */
    @Override
    public double predict(int userIdx, int itemIdx) throws LibrecException {
        //create userSimilarityList if not exists
        if (!(null != userSimilarityList && userSimilarityList.length > 0)) {
            createUserSimilarityList();
        }
        // find a number of similar users
        List<Map.Entry<Integer, Double>> nns = new ArrayList<>();
        List<Map.Entry<Integer, Double>> simList = userSimilarityList[userIdx];

        int count = 0;
        Set<Integer> userSet = trainMatrix.getRowsSet(itemIdx);
        for (Map.Entry<Integer, Double> userRatingEntry : simList) {
            int similarUserIdx = userRatingEntry.getKey();
            if (!userSet.contains(similarUserIdx)) {
                continue;
            }
            double sim = userRatingEntry.getValue();

            if (isRanking) {
                nns.add(userRatingEntry);
                count++;
            } else if (sim > 0) {
                nns.add(userRatingEntry);
                count++;
            }
            if (count == knn) {
                break;
            }
        }
        if (nns.size() == 0) {
            return isRanking ? 0 : globalMean;
        }
        if (isRanking) {
            double sum = 0.0d;
            for (Entry<Integer, Double> userRatingEntry : nns) {
                sum += userRatingEntry.getValue();
            }
            return sum;
        } else {
            // for rating prediction
            double sum = 0, ws = 0;
            for (Entry<Integer, Double> userRatingEntry : nns) {
                int similarUserIdx = userRatingEntry.getKey();
                double sim = userRatingEntry.getValue();
                double rate = trainMatrix.get(similarUserIdx, itemIdx);
                sum += sim * (rate - userMeans.get(similarUserIdx));
                ws += Math.abs(sim);
            }
            return ws > 0 ? userMeans.get(userIdx) + sum / ws : globalMean;
        }
    }

    /**
     * Create userSimilarityList.
     */
    public void createUserSimilarityList() {

        userSimilarityList = new ArrayList[numUsers];
        // these lines were used to make sure the similarity list
        // was being created on the new similarities, after
        // significance weighting was applied
        // System.out.println(similarityMatrix.get(1,0));
        // System.out.println(similarityMatrix.get(0,1));

        for (int userIndex = 0; userIndex < numUsers; ++userIndex) {
            SparseVector similarityVector = similarityMatrix.row(userIndex);
            userSimilarityList[userIndex] = new ArrayList<>(similarityVector.size());
            Iterator<VectorEntry> simItr = similarityVector.iterator();
            while (simItr.hasNext()) {
                VectorEntry simVectorEntry = simItr.next();

                userSimilarityList[userIndex].add(new AbstractMap.SimpleImmutableEntry<>(simVectorEntry.index(), simVectorEntry.get()));
            }
            Lists.sortList(userSimilarityList[userIndex], false);
        }
    }
}
