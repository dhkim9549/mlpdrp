package com.dhkim9549.mlpdrp;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import scala.util.Random;

import java.io.*;

/**
 * Evaluate a trained MLP DRP model
 */
public class MLPDRPEval {

    static LineNumberReader in = null;
    static String testDataInputFileName = "/down/collect_data/collect_data_20150101.txt";

    public static void main(String[] args) throws Exception {

        MultiLayerNetwork model = MLPDRP.readModelFromFile("/down/sin/drp_model_MLPDRP_h2_uSGD_mb16_ss16_650000.zip");

        evaluateModelBatch(model);

    }

    public static void evaluateModelBatch(MultiLayerNetwork model) throws Exception {

        System.out.println("Evaluating batch...");

        // Training data input file reader
        in = new LineNumberReader(new FileReader(testDataInputFileName));

        // Evaluation result output writer
        BufferedWriter out = new BufferedWriter(new FileWriter("/down/drp_list_eval.txt"));

        String header = "";
        header += "seq\t";
        header += "guarnt_no\t";
        header += "cllct_rate\t";
        header += "predicted_cllct_rate\t";
        header += "cllct_rate_old\t";
        header += "debt_ramt\t";
        header += "dischrg_dur_month\t";
        out.write(header + "\n");
        out.flush();

        int i = 0;
        Random rand = new Random();

        String s = "";
        while((s = in.readLine()) != null) {

            i++;
            if(i % 10000 == 0) {
                System.out.println("i = " + i);
            }

            if(s.indexOf("CLLCT_RATE") >= 0) {
                continue;
            }

            String guarnt_no = MLPDRP.getToken(s, 1, "\t");
            double cllct_rate = Double.parseDouble(MLPDRP.getToken(s, 18, "\t"));
            double cllct_rate_old = Double.parseDouble(MLPDRP.getToken(s, 17, "\t"));
            long debt_ramt = Long.parseLong(MLPDRP.getToken(s, 16, "\t"));
            long dischrg_dur_month = Long.parseLong(MLPDRP.getToken(s, 3, "\t"));
            long seq = rand.nextLong();

            double[] featureData = new double[MLPDRP.numOfInputs];

            featureData[0] = cllct_rate_old;
            featureData[1] = MLPDRP.rescaleAmt(debt_ramt);
            featureData[2] = MLPDRP.rescaleAmt(dischrg_dur_month, 0, 120);

            INDArray feature = Nd4j.create(featureData, new int[]{1, MLPDRP.numOfInputs});
            INDArray output = model.output(feature);

            double predicted_cllct_rat = output.getDouble(0);

            /*
            System.out.print("feature = " + feature);
            System.out.print("  output = " + output);
            System.out.println("  cllct_rate = " + cllct_rate);
            */

            String s2 = "";
            s2 += seq + "\t";
            s2 += guarnt_no + "\t";
            s2 += cllct_rate + "\t";
            s2 += predicted_cllct_rat + "\t";
            s2 += cllct_rate_old + "\t";
            s2 += debt_ramt + "\t";
            s2 += dischrg_dur_month + "\t";

            out.write(s2 + "\n");
            out.flush();
        }

        out.close();
    }
}