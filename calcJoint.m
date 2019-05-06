function p_joint = calcJoint( i, j, probabilities, strategies)

       p_00= sum(probabilities(strategies(:, i) == 0 & strategies(:, j) == 0)) ;
       p_01= sum(probabilities(strategies(:, i) == 0 & strategies(:, j) == 1)) ;
       p_10= sum(probabilities(strategies(:, i) == 1 & strategies(:, j) == 0));
       p_11= sum(probabilities(strategies(:, i) == 1 & strategies(:, j) == 1));
       
        p_joint = [p_00 p_01 p_10 p_11];
        
        p_joint(p_joint == 0) = 10e-10;
        
    end