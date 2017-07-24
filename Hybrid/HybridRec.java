package net.librec.recommender.content;
import net.librec.recommender.NaiveBayesRecommender;
import net.librec.annotation.ModelData;
import net.librec.common.LibrecException;
import net.librec.recommender.AbstractRecommender;
import net.librec.recommender.UserKNNRec;


@ModelData({"isRanking", "trainMatrix"})
public class HybridRec extends AbstractRecommender

{

    // make our bayes rec and knn collabroative rec
    NaiveBayesRecommender NBRec;
    UserKNNRec UKRec;
    double weight_nb;
    double weight_cf;


    @Override
    protected void setup() throws LibrecException {

        // must set the context for each of the recs and run the setup method which was changed to public
        super.setup();
        context = getContext();

        NBRec = new NaiveBayesRecommender();
        NBRec.setContext(context);
        NBRec.setup();

        UKRec = new UserKNNRec();
        UKRec.setContext(context);
        UKRec.setup();

    }

    protected void trainModel () throws LibrecException{

        // train each model, which was made a public method in both files
        NBRec.trainModel();
        UKRec.trainModel();

    }

    public double predict (int user, int item) throws LibrecException {

        // get weights for making hybrid prediction
        weight_nb = conf.getDouble("weight.nb");
        weight_cf = conf.getDouble("weight.cf");

        // call predict and make the aggregated prediction
        double NBpred = NBRec.predict(user, item);
        double UKpred = UKRec.predict(user, item);
        double agg = (weight_nb * NBpred) + (weight_cf * UKpred);

        // return the results
        //System.out.println(agg);
        return agg;

    }


}