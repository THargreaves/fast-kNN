import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.io.*;
import java.lang.*;

public class java_naive_iris {

  public static void main(String[] args) throws FileNotFoundException, IOException {

    List<List<String>> records = new ArrayList<>();
    try (BufferedReader br = new BufferedReader(new FileReader("iris.csv"))) {
        String line;
        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            records.add(Arrays.asList(values));
        }
    }

    int[] test_inds = {61,36,139,144,95,78,77,15,14,105,133,33,110,101,126,74,
                       91,62,16,20,128,25,106,55,31,112,53,149,114,92,143,104,
                       141,24,138,97,7,79,39,122,64,123,47,116,80,17,34,32,98,136};

    double[][] iris = new double[records.size()-1][5];
    for(int i=0; i<records.size()-1; i++){
      for(int j=0; j<5; j++) {
        //System.out.print((records.get(i)).get(j).getClass().getName());
        iris[i][j] = Double.valueOf((records.get(i+1)).get(j)); // i+1 since row 0 is column headers
      }
    }

    int[] train_inds = new int[0];
    for(int i=0; i<iris.length; i++) {

      boolean found = false;

      for (int n : test_inds) {
        if (n == i) {
          found = true;
          break;
        }
      }

      if(!found) {
        int[] temp = new int[train_inds.length + 1];
        for(int e=0; e<train_inds.length; e++) {
          temp[e] = train_inds[e];
        }
        temp[train_inds.length] = i;
        train_inds = temp;
      }
    }

    double[][] Xtt = new double[test_inds.length][4];
    double[] Ytt = new double[test_inds.length];
    for(int i=0; i<test_inds.length; i++) {
      Xtt[i] = iris[test_inds[i]];
      Ytt[i] = iris[test_inds[i]][4];
    }

    double[][] Xtr = new double[train_inds.length][4];
    double[] Ytr = new double[train_inds.length];
    for(int i=0; i<train_inds.length; i++) {
      Xtr[i] = iris[train_inds[i]];
      Ytr[i] = iris[train_inds[i]][4];
    }

    Naive kNN = new Naive(Xtr, Ytr);
    double[] preds = kNN.predict(Xtt, 5);

    System.out.print(Arrays.toString(preds));

  }
}
