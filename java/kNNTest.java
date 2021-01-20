import java.util.Random;
import java.util.Arrays;

public class kNNTest {

  public static void main(String[] args) {

    int seed = 42; // set seed for rng for reproducibility

    int N = 10; // number of training data points
    int M = 4; // number of test data points
    int d = 2; // dimension of data points
    int k = 3; // number of neighbours
    int p = 2; // number of classes

    double[][] Xtr = new double[N][d]; // training data
    int[] Ytr = new int[N]; // training labels

    double[][] Xtt = new double[M][d]; // test data

    Random rng = new Random(seed); // call random number generator

    // The next two blocks of for loops generate the random data

    for(int i = 0; i < N; i++){
      for(int j = 0; j < d; j++) {
        Xtr[i][j] = rng.nextGaussian(); // generate random data points
      }
      Ytr[i] = rng.nextInt(p); // generate classes for those data points
    }

    for(int i = 0; i < M; i++) {
      for(int j = 0; j < d; j++) {
        Xtr[i][j] = rng.nextGaussian(); // generate random data points
      }
    }

    kNN kNNtest1 = new kNN(); // create new classifier
    kNNtest1.fit(Xtr, Ytr); // fit the classifier with training data

    System.out.print(Arrays.toString(kNNtest1.predict(k, Xtt))); // predictions

    // The results from that print should match those from below

    kNN kNNtest2 = new kNN(Xtr, Ytr); // try fitting on creation
    System.out.print(Arrays.toString(kNNtest2.predict(k, Xtt))); // predictions

  }
}
