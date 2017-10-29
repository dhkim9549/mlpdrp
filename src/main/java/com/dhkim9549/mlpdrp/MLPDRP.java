package com.dhkim9549.mlpdrp;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.*;
import java.io.*;

/**
 *  Building a Debt Repayment Prediction Model with MLP
 * @author Dong-Hyun Kim
 */
public class MLPDRP {

    static String hpId = "MLPDRP_h3_uSGD_mb16_ss16";

    //double learnigRate = Double.parseDouble(args[0]);
    static double learnigRate = 0.0025;

    // Number of sample size per iteration
    static long nSamples = 16;

    // Mini-batch size
    static int batchSize = 16;

    // Evaluation sample size
    static long nEvalSamples = 10000;

    // Number of input variables to the neural network
    static int numOfInputs = 19;

    static LineNumberReader in = null;
    static String trainingDataInputFileName = "/down/collect_data/collect_data_20130101.txt";

    public static void main(String[] args) throws Exception {

        System.out.println("************************************************");
        System.out.println("hpId = " + hpId);
        System.out.println("Number of hidden layers = 3");
        System.out.println("learnigRate = " + learnigRate);
        System.out.println("Updater = " + "SGD");
        System.out.println("mini-batch size (batchSize) = " + batchSize);
        System.out.println("Number of sample size per iteration (nSamples) = " + nSamples);
        System.out.println("i >= 0");
        System.out.println("************************************************");

        MultiLayerNetwork model = getInitModel(learnigRate);
        //MultiLayerNetwork model = readModelFromFile("/down/sin/css_model_MLPDRP_h2_uSGD_mb16_ss16_200000.zip");

        NeuralNetConfiguration config = model.conf();
        System.out.println("config = " + config);

        // Training data input file reader
        in = new LineNumberReader(new FileReader(trainingDataInputFileName));

        // Training iteration
        long i = 0;

        while(true) {

            i++;

            if(i % 10000 == 0) {
                System.out.println("i = " + i);
                // evaluateModel(model);
            }

            if(i % 50000 == 0) {
                MLPDRPEval.evaluateModelBatch(model);
            }

            List<DataSet> listDs = getTrainingData();
            DataSetIterator trainIter = new ListDataSetIterator(listDs, batchSize);

            // Train the model
            model = train(model, trainIter);

            if (i % 50000 == 0) {
                writeModelToFile(model, "/down/drp_model_" + hpId + "_" + i + ".zip");
            }
        }
    }

    public static MultiLayerNetwork getInitModel(double learningRate) throws Exception {

        int seed = 123;

        int numInputs = numOfInputs;
        int numOutputs = 2;
        int numHiddenNodes = 30;

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.SGD)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        return model;
    }

    public static MultiLayerNetwork train(MultiLayerNetwork model, DataSetIterator trainIter) throws Exception {

        //model.setListeners(new ScoreIterationListener(1000));

        model.fit( trainIter );

        return model;
    }

    private static List<DataSet> getTrainingData() throws Exception {

        //System.out.println("Getting training data...");

        List<DataSet> listDs = new ArrayList<>();

        while(listDs.size() < nSamples) {

            String s = in.readLine();
            if(s == null) {
                System.out.println("Training data file rollover...");
                in.close();
                in = new LineNumberReader(new FileReader(trainingDataInputFileName));
                continue;
            }
            if(s.indexOf("CLLCT_RATE") >= 0) {
                continue;
            }

            DataSet ds = getDataSet(s);
            listDs.add(ds);
        }

        Collections.shuffle(listDs);

        //System.out.println("listDs.size() = " + listDs.size());
        //System.out.println("Getting training data complete.");

        return listDs;
    }

    public static String getToken(String s, int x) {
        return getToken(s, x, " \t\n\r\f");
    }

    public static String getToken(String s, int x, String delim) {

        s = s.replaceAll("\t", "\t ");

        StringTokenizer st = new StringTokenizer(s, delim);
        int counter = 0;
        String answer = null;
        while(st.hasMoreTokens()) {
            String token = st.nextToken();
            if(counter == x) {
                answer = token.trim();
            }
            counter++;
        }
        return answer;
    }

    private static DataSet getDataSet(String s) throws Exception {

        String guarnt_no = getToken(s, 1, "\t");
        double cllct_rate = Double.parseDouble(getToken(s, 18, "\t"));
        double cllct_rate_old = Double.parseDouble(getToken(s, 17, "\t"));
        long debt_ramt = Long.parseLong(getToken(s, 16, "\t"));
        long dischrg_dur_month = Long.parseLong(getToken(s, 3, "\t"));
        long org_guarnt_dur_month = Long.parseLong(getToken(s, 2, "\t"));
        String guarnt_dvcd_rent_yn = getToken(s, 4, "\t");
        String guarnt_dvcd_mid_yn = getToken(s, 5, "\t");
        String guarnt_dvcd_buy_yn = getToken(s, 6, "\t");
        String crdrc_yn = getToken(s, 7, "\t");
        String revivl_yn = getToken(s, 8, "\t");
        String exempt_yn = getToken(s, 9, "\t");
        String sptrepay_yn = getToken(s, 10, "\t");
        String psvact_yn = getToken(s, 11, "\t");
        long rdbtr_1_cnt = Long.parseLong(getToken(s, 12, "\t")); // new input
        long rdbtr_2_cnt = Long.parseLong(getToken(s, 13, "\t")); // new input
        long age = Long.parseLong(getToken(s, 14, "\t")); // new input
        long dischrg_occr_amt = Long.parseLong(getToken(s, 15, "\t")); // new input
        String prscp_cmplt_yn = getToken(s, 19, "\t"); // new input
        String ibon_amtz_yn = getToken(s, 20, "\t"); // new input
        long rdbtr_3_cnt = Long.parseLong(getToken(s, 21, "\t")); // new input

        double[] featureData = new double[numOfInputs];
        double[] labelData = new double[2];

        featureData[0] = cllct_rate_old;
        featureData[1] = rescaleAmt(debt_ramt);
        featureData[2] = rescaleAmt(dischrg_dur_month, 0, 120);
        featureData[3] = rescaleAmt(org_guarnt_dur_month, 0, 120, true);
        featureData[4] = rescaleYn(guarnt_dvcd_rent_yn);
        featureData[5] = rescaleYn(guarnt_dvcd_mid_yn);
        featureData[6] = rescaleYn(guarnt_dvcd_buy_yn);
        featureData[7] = rescaleYn(crdrc_yn);
        featureData[8] = rescaleYn(revivl_yn);
        featureData[9] = rescaleYn(exempt_yn);
        featureData[10] = rescaleYn(sptrepay_yn);
        featureData[11] = rescaleYn(psvact_yn);
        featureData[12] = rescaleNum(rdbtr_1_cnt); // new input
        featureData[13] = rescaleNum(rdbtr_2_cnt); // new input
        featureData[14] = rescaleAmt(age, 0, 100); // new input
        featureData[15] = rescaleAmt(dischrg_occr_amt); // new input
        featureData[16] = rescaleYn(prscp_cmplt_yn); // new input
        featureData[17] = rescaleYn(ibon_amtz_yn); // new input
        featureData[18] = rescaleNum(rdbtr_3_cnt); // new input

        labelData[0] = cllct_rate;
        labelData[1] = 1.0 - cllct_rate;

        INDArray feature = Nd4j.create(featureData, new int[]{1, numOfInputs});
        INDArray label = Nd4j.create(labelData, new int[]{1, 2});

        DataSet ds = new DataSet(feature, label);

        /*
        System.out.println("\n guarnt_no = " + guarnt_no);
        System.out.println(rdbtr_2_cnt + " " + age + " " + dischrg_occr_amt + " " + prscp_cmplt_yn + " " + ibon_amtz_yn);
        System.out.println(cllct_rate);
        System.out.println("ds = " + ds);
        */

        return ds;
    }

    public static void evaluateModel(MultiLayerNetwork model) {

        System.out.println("Evaluating...");

        for(int i = 0; i <= 10; i++) {
            double[] featureData = new double[numOfInputs];
            featureData[0] = 0.0;
            featureData[1] = rescaleAmt(20000000);
            featureData[2] = rescaleAmt(12 * 5, 0, 120);
            featureData[3] = rescaleAmt(36, 0, 120);
            featureData[4] = 1.0;
            featureData[5] = 0.0;
            featureData[6] = 0.0;
            featureData[7] = 0.0;
            featureData[8] = 0.0;
            featureData[9] = 0.0;
            featureData[10] = 0.0;
            featureData[11] = 0.0;
            featureData[12] = 0.0;; // new input
            featureData[13] = 0.0; // new input
            featureData[14] = rescaleAmt(40, 0, 100); // new input
            featureData[15] = rescaleAmt(20000000); // new input
            featureData[16] = rescaleYn("N"); // new input
            featureData[17] = 0.0; // new input
            featureData[18] = 0.1 * i; // new input

            INDArray feature = Nd4j.create(featureData, new int[]{1, numOfInputs});
            INDArray output = model.output(feature);
            System.out.print("feature = " + feature);
            System.out.print("  output = " + output);
            double cllct_rate = output.getDouble(0);
            System.out.println("  predicted cllct_rate = " + cllct_rate);
        }
    }

    public static double rescaleAmt(long x) {
        return rescaleAmt(x, 0, 100000000);
    }

    public static double rescaleAmt(double x, double min, double max) {
        return rescaleAmt(x, min, max, false);
    }

    public static double rescaleAmt(double x, double min, double max, boolean forceMin) {
        if(forceMin) {
            if(x < min) {
                x = min;
            }
        }
        double base = (max - min) / 10.0;
        double y = (Math.log(x - min + base) - Math.log(base)) / (Math.log(max - min + base) - Math.log(base));
        return y;
    }

    public static double rescaleYn(String x) {
        double y = 0.0;
        if(x.equals("Y")) {
            y = 1.0;
        }
        return y;
    }

    public static double rescaleNum(long x) {
        double y = 0.0;
        if(x > 0) {
            y = 1.0;
        }
        return y;
    }

    public static MultiLayerNetwork readModelFromFile(String fileName) throws Exception {

        System.out.println("Deserializing model...");

        // Load the model
        File locationToSave = new File(fileName);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        System.out.println("Deserializing model complete.");

        return model;
    }

    public static void writeModelToFile(MultiLayerNetwork model, String fileName) throws Exception {

        System.out.println("Serializing model...");

        // Save the model
        File locationToSave = new File(fileName); // Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true; // Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);

        System.out.println("Serializing model complete.");

    }
}