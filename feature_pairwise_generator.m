


function feature_pairwise=feature_pairwise_generator(nodes,w2vFeatures,general)
%gt=1 the general pairwise features assuming or labeld 1
%we assume feature_pairwise is additive and is non-zero only between the
%samples which have different nodes.
global n_nodes;
global n_word2vec_features;
global n_pairs;
feature_pairwise=zeros(n_nodes,n_nodes,n_word2vec_features);
for i=1:n_nodes
    for j=1:n_nodes
        if (i==j)
            feature_pairwise(i,j,:)=0;
        else
            %if(i<j)
            if ((nodes(i)~=nodes(j))|| (general==1))
                feature_pairwise(i,j,:)= 1./(abs(w2vFeatures(:,i)- w2vFeatures(:,j))+1);
            end
            %   end
        end
    end
end


end



