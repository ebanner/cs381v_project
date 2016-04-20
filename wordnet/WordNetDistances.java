import java.math.RoundingMode;
import java.text.DecimalFormat;

import edu.cmu.lti.lexical_db.ILexicalDatabase;
import edu.cmu.lti.lexical_db.NictWordNet;
import edu.cmu.lti.ws4j.RelatednessCalculator;
import edu.cmu.lti.ws4j.impl.HirstStOnge;
import edu.cmu.lti.ws4j.impl.JiangConrath;
import edu.cmu.lti.ws4j.impl.LeacockChodorow;
import edu.cmu.lti.ws4j.impl.Lesk;
import edu.cmu.lti.ws4j.impl.Lin;
import edu.cmu.lti.ws4j.impl.Path;
import edu.cmu.lti.ws4j.impl.Resnik;
import edu.cmu.lti.ws4j.impl.WuPalmer;
import edu.cmu.lti.ws4j.util.WS4JConfiguration;


public class WordNetDistances {

  private static ILexicalDatabase db = new NictWordNet();

  // Available options of metrics.
  private static RelatednessCalculator[] metrics = {
      new ZhaoEtAl(db),   // from Zhao et al. paper
      new Path(db),       // generic path metric
      new WuPalmer(db)   // common metric, Budanitsky & Hirst review, 2.5.2
      //new HirstStOnge(db),      // ?
      //new Lesk(db),             // ?
      //new LeacockChodorow(db),  // Budanitsky & Hirst review, 2.5.3
      //new Resnik(db),           // Budanitsky & Hirst review, 2.6.1
      //new JiangConrath(db)     // Budanitsky & Hirst review, 2.6.2
      //new Lin(db)               // Budanitsky & Hirst review, 2.6.3
  };
  private static String[] metricNames = {
      "ZHAO_ET_AL",
      "PATH",
      "WUP"
      //"HIRST",
      //"LESK",
      //"LEACOCK_CHODOROW",
      //"RESNIK",
      //"JIANG_CONRATH"
      //"LIN"
  };

  private static double compute(String word1, String word2,
                                RelatednessCalculator rc) {
    WS4JConfiguration.getInstance().setMFS(true);
    double s = rc.calcRelatednessOfWords(word1, word2);
    return s;
  }

  // Pass in words as arguments.
  public static void main(String[] args) {
    //// In case you want to round the decimal values:
    //DecimalFormat df = new DecimalFormat("#.##########");
    //df.setRoundingMode(RoundingMode.CEILING);
    // Loop through all the words.
    for (int m = 0; m < metrics.length; m++) {
      System.out.println("-----METRIC = " + metricNames[m] + "-----");
      for (int i = 0; i < args.length; i++) {
        System.out.println("# " + args[i]);
        for (int j = 0; j < args.length; j++) {
          double distance = 1;
          // Distance stays at 1 if it's the same word because otherwise it blows
          // with on some metrics.
          if (i != j) {
            distance = compute(args[i], args[j], metrics[m]);
          }
          //System.out.print(df.format(distance) + " ");
          System.out.print(distance + " ");
        }
        System.out.println();
      }
      System.out.println();
    }
  }

}
