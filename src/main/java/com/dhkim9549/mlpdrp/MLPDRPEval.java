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
    static String modelInputFileName = "/down/sin/drp_model_MLPDRP_h2_uSGD_mb16_ss16_150000.zip";

    public static void main(String[] args) throws Exception {

        if(args.length >= 1) {
            modelInputFileName = args[0];
        }

        MultiLayerNetwork model = MLPDRP.readModelFromFile(modelInputFileName);

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
        double squared_error = 0.0;

        String s = "";
        while((s = in.readLine()) != null) {

            i++;
            if(i % 10000 == 0) {
                System.out.println("i = " + i);
                System.out.println("MSE = " + (squared_error / (double)i));
            }

            if(s.indexOf("CLLCT_RATE") >= 0) {
                continue;
            }

            long seq = rand.nextLong();
            String guarnt_no = MLPDRP.getToken(s, 1, "\t");
            double cllct_rate = Double.parseDouble(MLPDRP.getToken(s, 18, "\t"));
            double cllct_rate_old = Double.parseDouble(MLPDRP.getToken(s, 17, "\t"));
            long debt_ramt = Long.parseLong(MLPDRP.getToken(s, 16, "\t"));
            long dischrg_dur_month = Long.parseLong(MLPDRP.getToken(s, 3, "\t"));
            long org_guarnt_dur_month = Long.parseLong(MLPDRP.getToken(s, 2, "\t")); // new input
            String guarnt_dvcd_rent_yn = MLPDRP.getToken(s, 4, "\t");
            String guarnt_dvcd_mid_yn = MLPDRP.getToken(s, 5, "\t");
            String guarnt_dvcd_buy_yn = MLPDRP.getToken(s, 6, "\t");
            String crdrc_yn = MLPDRP.getToken(s, 7, "\t");
            String revivl_yn = MLPDRP.getToken(s, 8, "\t");
            String exempt_yn = MLPDRP.getToken(s, 9, "\t");
            String sptrepay_yn = MLPDRP.getToken(s, 10, "\t");
            String psvact_yn = MLPDRP.getToken(s, 11, "\t");
            long rdbtr_1_cnt = Long.parseLong(MLPDRP.getToken(s, 12, "\t")); // new input
            long rdbtr_2_cnt = Long.parseLong(MLPDRP.getToken(s, 13, "\t")); // new input
            long age = Long.parseLong(MLPDRP.getToken(s, 14, "\t")); // new input
            long dischrg_occr_amt = Long.parseLong(MLPDRP.getToken(s, 15, "\t")); // new input
            String prscp_cmplt_yn = MLPDRP.getToken(s, 19, "\t"); // new input
            String ibon_amtz_yn = MLPDRP.getToken(s, 20, "\t"); // new input


            double[] featureData = new double[MLPDRP.numOfInputs];

            featureData[0] = cllct_rate_old;
            featureData[1] = MLPDRP.rescaleAmt(debt_ramt);
            featureData[2] = MLPDRP.rescaleAmt(dischrg_dur_month, 0, 120);
            featureData[3] = MLPDRP.rescaleAmt(org_guarnt_dur_month, 0, 120, true);
            featureData[4] = MLPDRP.rescaleYn(guarnt_dvcd_rent_yn);
            featureData[5] = MLPDRP.rescaleYn(guarnt_dvcd_mid_yn);
            featureData[6] = MLPDRP.rescaleYn(guarnt_dvcd_buy_yn);
            featureData[7] = MLPDRP.rescaleYn(crdrc_yn);
            featureData[8] = MLPDRP.rescaleYn(revivl_yn);
            featureData[9] = MLPDRP.rescaleYn(exempt_yn);
            featureData[10] = MLPDRP.rescaleYn(sptrepay_yn);
            featureData[11] = MLPDRP.rescaleYn(psvact_yn);
            /*
            featureData[12] = MLPDRP.rescaleNum(rdbtr_1_cnt); // new input
            featureData[13] = MLPDRP.rescaleNum(rdbtr_2_cnt); // new input
            featureData[14] = MLPDRP.rescaleAmt(age, 0, 100); // new input
            featureData[15] = MLPDRP.rescaleAmt(dischrg_occr_amt); // new input
            featureData[16] = MLPDRP.rescaleYn(prscp_cmplt_yn); // new input
            featureData[17] = MLPDRP.rescaleYn(ibon_amtz_yn); // new input
            */

            INDArray feature = Nd4j.create(featureData, new int[]{1, MLPDRP.numOfInputs});
            INDArray output = model.output(feature);

            double predicted_cllct_rat = output.getDouble(0);

            squared_error +=  Math.pow((predicted_cllct_rat - cllct_rate), 2.0);

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
            s2 += org_guarnt_dur_month + "\t";
            s2 += guarnt_dvcd_rent_yn + "\t";

            out.write(s2 + "\n");
            out.flush();
        }

        System.out.println("MSE = " + (squared_error / (double)i));

        out.close();
    }
}