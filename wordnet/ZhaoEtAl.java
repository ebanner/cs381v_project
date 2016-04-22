import java.lang.Math;
import java.util.ArrayList;
import java.util.List;

import edu.cmu.lti.jawjaw.pobj.POS;
import edu.cmu.lti.lexical_db.ILexicalDatabase;
import edu.cmu.lti.lexical_db.data.Concept;
import edu.cmu.lti.ws4j.Relatedness;
import edu.cmu.lti.ws4j.RelatednessCalculator;
import edu.cmu.lti.ws4j.util.PathFinder.Subsumer;
import edu.cmu.lti.ws4j.util.DepthFinder.Depth;

public class ZhaoEtAl extends RelatednessCalculator {

	protected static double min = 0; // actually, (0, 1]
	protected static double max = 1;
	
	@SuppressWarnings("serial")
	private static List<POS[]> posPairs = new ArrayList<POS[]>(){{
		add(new POS[]{POS.n,POS.n});
		add(new POS[]{POS.v,POS.v});
	}};

	public ZhaoEtAl(ILexicalDatabase db) {
		super(db);
	}

	protected Relatedness calcRelatedness( Concept synset1, Concept synset2 ) {
		StringBuilder tracer = new StringBuilder();
		if ( synset1 == null || synset2 == null ) return new Relatedness( min, null, illegalSynset );
		if ( synset1.getSynset().equals( synset2.getSynset() ) ) return new Relatedness( max, identicalSynset, null );
		
		StringBuilder subTracer = enableTrace ? new StringBuilder() : null;

		List<Depth> lcsList = depthFinder.getRelatedness( synset1, synset2, subTracer );
		if ( lcsList.size() == 0 ) return new Relatedness( min );
    // Depth of lowest common super-ordinate.
		int depth_lcs = lcsList.get(0).depth; // sorted by depth (asc)

    // Get max depth of the two paths.
		int depth1 = depthFinder.getShortestDepth( synset1 );
		int depth2 = depthFinder.getShortestDepth( synset2 );
    int max_depth = Math.max(depth1, depth2);

    double score;
    if (max_depth > 0) {
      score = (double)depth_lcs / (double)max_depth;
    } else {
      score = -1;
    }
		
		if ( enableTrace ) {
			tracer.append( subTracer.toString() );
			tracer.append( "LCS length: "+depth_lcs+"\n" );
			tracer.append( "Max length = "+max_depth+"\n" );
		}
				
		return new Relatedness( score, tracer.toString(), null );
	}
	
	@Override
	public List<POS[]> getPOSPairs() {
		return posPairs;
	}
}
