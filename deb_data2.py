# Updated data file
# clusters_data = {'Tauopathy': ['MAPT', 'METTL14', 'METTL3', 'Tau Protein', 'Tau Oligomers', 'Tau 25kD Fragment', 'N-224 Tau Fragment', "Alzheimer's Disease", 'Frontotemporal Dementia (FTD)', 'Tau Seeding', 'EIF4G2', 'MYT1', 'NSD1', 'NCOA6'], 'Autophagy': ['MTOR', 'WIPI2', 'CTSD', 'mTOR Signaling'], 'Ubiquitin/Proteasome': ['CUL5', 'SOCS4', 'SOCS5', 'ARIH2', 'RNF7', 'ELOB', 'ELOC', 'CRL5SOCS4 Complex', 'CHIP', 'FBXW7'], 'UFMylation': ['UFC1', 'UFM1', 'UFL1', 'UBA5'], 'Neddylation': ['UBE2F'], 'Proteasome Activation': ['PSME1', 'PSME2', 'PA28'], 'Oxidative Stress': ['KEAP1', 'ROS Response'], 'Mitochondrial Function': ['FECH', 'FH', 'Nuclear Mitochondrial Genes', 'Oxidative Phosphorylation', 'Electron Transport Chain', 'TCA Cycle'], 'Lysosomal Function': ['PSAP'], 'Gene Expression': ['RNA Transport', 'RNA Degradation'], 'Metabolic Regulation': ['AMPK Signaling'], 'Protein Modification': ['GPI-anchor Biosynthesis', 'UFMylation', 'Neddylation'], 'Immune Regulation': ['Neuro-immune Axis'], 'Protein Clearance': ['Autophagy', 'Ubiquitin/Proteasome System'], 'mTOR Pathway': ['TSC1', 'TSC2', 'RHEB', 'pS6'], 'Experimental System': ['SH-SY5Y cell line', 'PC1', 'SC2', 'FACS sorted low tau population', 'Cas9', 'SP70 antibody', 'gRNA MAPT-1', 'gRNA MAPT-2']}

# nodes_data = [{'id': 'MAPT', 'type': 'gene', 'cluster': 'Tauopathy', 'size': 3.22}, {'id': 'METTL14', 'type': 'gene', 'cluster': 'Tauopathy', 'size': 1.1}, {'id': 'METTL3', 'type': 'gene', 'cluster': 'Tauopathy', 'size': 1.1}, {'id': 'MTOR', 'type': 'gene', 'cluster': 'Autophagy', 'size': 1.61}, {'id': 'CUL5', 'type': 'gene', 'cluster': 'Ubiquitin/Proteasome', 'size': 2.71}, {'id': 'SOCS4', 'type': 'gene', 'cluster': 'Ubiquitin/Proteasome', 'size': 2.3}, {'id': 'SOCS5', 'type': 'gene', 'cluster': 'Ubiquitin/Proteasome', 'size': 1.1}, {'id': 'ARIH2', 'type': 'gene', 'cluster': 'Ubiquitin/Proteasome', 'size': 1.39}, {'id': 'RNF7', 'type': 'gene', 'cluster': 'Ubiquitin/Proteasome', 'size': 1.39}, {'id': 'UFC1', 'type': 'gene', 'cluster': 'UFMylation', 'size': 0.69}, {'id': 'UFM1', 'type': 'gene', 'cluster': 'UFMylation', 'size': 0.69}, {'id': 'UFL1', 'type': 'gene', 'cluster': 'UFMylation', 'size': 0.69}, {'id': 'UBA5', 'type': 'gene', 'cluster': 'UFMylation', 'size': 0.69}, {'id': 'ELOB', 'type': 'gene', 'cluster': 'Ubiquitin/Proteasome', 'size': 1.1}, {'id': 'ELOC', 'type': 'gene', 'cluster': 'Ubiquitin/Proteasome', 'size': 1.1}, {'id': 'UBE2F', 'type': 'gene', 'cluster': 'Neddylation', 'size': 0.69}, {'id': 'PSME1', 'type': 'gene', 'cluster': 'Proteasome Activation', 'size': 0.69}, {'id': 'PSME2', 'type': 'gene', 'cluster': 'Proteasome Activation', 'size': 0.69}, {'id': 'KEAP1', 'type': 'gene', 'cluster': 'Oxidative Stress', 'size': 0.69}, {'id': 'FECH', 'type': 'gene', 'cluster': 'Mitochondrial Function', 'size': 0.69}, {'id': 'FH', 'type': 'gene', 'cluster': 'Mitochondrial Function', 'size': 0.69}, {'id': 'WIPI2', 'type': 'gene', 'cluster': 'Autophagy', 'size': 0.69}, {'id': 'CTSD', 'type': 'gene', 'cluster': 'Autophagy', 'size': 0.69}, {'id': 'PSAP', 'type': 'gene', 'cluster': 'Lysosomal Function', 'size': 0.69}, {'id': 'Nuclear Mitochondrial Genes', 'type': 'gene group', 'cluster': 'Mitochondrial Function', 'size': 0.69}, {'id': 'Tau Protein', 'type': 'protein', 'cluster': 'Tauopathy', 'size': 3.4}, {'id': 'Tau Oligomers', 'type': 'protein', 'cluster': 'Tauopathy', 'size': 2.71}, {'id': 'CRL5SOCS4 Complex', 'type': 'protein complex', 'cluster': 'Ubiquitin/Proteasome', 'size': 2.3}, {'id': 'CHIP', 'type': 'protein', 'cluster': 'Ubiquitin/Proteasome', 'size': 1.39}, {'id': 'PA28', 'type': 'protein complex', 'cluster': 'Proteasome Activation', 'size': 0.69}, {'id': 'Tau 25kD Fragment', 'type': 'protein fragment', 'cluster': 'Tauopathy', 'size': 2.08}, {'id': 'N-224 Tau Fragment', 'type': 'protein fragment', 'cluster': 'Tauopathy', 'size': 0.69}, {'id': 'Calpain', 'type': 'protein', 'cluster': 'Protease', 'size': 0.69}, {'id': "Alzheimer's Disease", 'type': 'disease', 'cluster': 'Tauopathy', 'size': 2.71}, {'id': 'Frontotemporal Dementia (FTD)', 'type': 'disease', 'cluster': 'Tauopathy', 'size': 2.3}, {'id': 'Oxidative Phosphorylation', 'type': 'pathway', 'cluster': 'Mitochondrial Function', 'size': 1.61}, {'id': 'Electron Transport Chain', 'type': 'pathway', 'cluster': 'Mitochondrial Function', 'size': 1.39}, {'id': 'Autophagy', 'type': 'pathway', 'cluster': 'Protein Clearance', 'size': 2.3}, {'id': 'Ubiquitin/Proteasome System', 'type': 'pathway', 'cluster': 'Protein Clearance', 'size': 2.08}, {'id': 'mTOR Signaling', 'type': 'pathway', 'cluster': 'Autophagy', 'size': 1.79}, {'id': 'GPI-anchor Biosynthesis', 'type': 'pathway', 'cluster': 'Protein Modification', 'size': 1.1}, {'id': 'UFMylation', 'type': 'pathway', 'cluster': 'Protein Modification', 'size': 1.61}, {'id': 'RNA Transport', 'type': 'pathway', 'cluster': 'Gene Expression', 'size': 1.1}, {'id': 'TCA Cycle', 'type': 'pathway', 'cluster': 'Mitochondrial Function', 'size': 0.69}, {'id': 'AMPK Signaling', 'type': 'pathway', 'cluster': 'Metabolic Regulation', 'size': 0.69}, {'id': 'RNA Degradation', 'type': 'pathway', 'cluster': 'Gene Expression', 'size': 0.69}, {'id': 'Neddylation', 'type': 'pathway', 'cluster': 'Protein Modification', 'size': 1.39}, {'id': 'ROS Response', 'type': 'pathway', 'cluster': 'Oxidative Stress', 'size': 1.79}, {'id': 'Tau Seeding', 'type': 'process', 'cluster': 'Tauopathy', 'size': 1.1}, {'id': 'Neuro-immune Axis', 'type': 'pathway', 'cluster': 'Immune Regulation', 'size': 0.69}, {'id': 'TSC1', 'type': 'gene', 'cluster': 'mTOR Pathway', 'size': 1.1}, {'id': 'Progressive Supranuclear Palsy', 'type': 'disease', 'cluster': 'Tauopathy', 'size': 0.69}, {'id': 'Corticobasal Degeneration', 'type': 'disease', 'cluster': 'Tauopathy', 'size': 0.69}, {'id': 'Pick�s Disease', 'type': 'disease', 'cluster': 'Tauopathy', 'size': 0.69}, {'id': 'FTDP-17', 'type': 'disease', 'cluster': 'Tauopathy', 'size': 0.69}, {'id': 'Chromatin Modification', 'type': 'pathway', 'cluster': 'Chromatin Modification', 'size': 0.69}, {'id': 'EIF4G2', 'type': 'gene', 'cluster': 'Tauopathy', 'size': 1.0}, {'id': 'MYT1', 'type': 'gene', 'cluster': 'Tauopathy', 'size': 1.0}, {'id': 'NSD1', 'type': 'gene', 'cluster': 'Tauopathy', 'size': 1.0}, {'id': 'FBXW7', 'type': 'gene', 'cluster': 'Ubiquitin/Proteasome', 'size': 1.0}, {'id': 'NCOA6', 'type': 'gene', 'cluster': 'Tauopathy', 'size': 1.0}, {'id': 'TSC2', 'type': 'gene', 'cluster': 'mTOR Pathway', 'size': 1.1}, {'id': 'RHEB', 'type': 'gene', 'cluster': 'mTOR Pathway', 'size': 0.8}, {'id': 'pS6', 'type': 'protein', 'cluster': 'mTOR Pathway', 'size': 0.8}, {'id': 'SH-SY5Y cell line', 'type': 'cell line', 'cluster': 'Experimental System', 'size': 2.3}, {'id': 'PC1', 'type': 'cell line clone', 'cluster': 'Experimental System', 'size': 1.61}, {'id': 'SC2', 'type': 'cell line clone', 'cluster': 'Experimental System', 'size': 1.61}, {'id': 'FACS sorted low tau population', 'type': 'cell population', 'cluster': 'Experimental System', 'size': 1.79}, {'id': 'Cas9', 'type': 'reagent', 'cluster': 'Experimental System', 'size': 2.08}, {'id': 'SP70 antibody', 'type': 'reagent', 'cluster': 'Experimental System', 'size': 1.39}, {'id': 'gRNA MAPT-1', 'type': 'reagent', 'cluster': 'Experimental System', 'size': 1.79}, {'id': 'gRNA MAPT-2', 'type': 'reagent', 'cluster': 'Experimental System', 'size': 1.79}]

# edges_data = [{'source': 'MAPT', 'target': 'Tau Protein', 'relation': 'encodes', 'score': 0.99}, {'source': 'CUL5', 'target': 'Tau Protein', 'relation': 'ubiquitinates', 'score': 0.75}, {'source': 'SOCS4', 'target': 'Tau Protein', 'relation': 'recruits for ubiquitination', 'score': 0.7}, {'source': 'SOCS5', 'target': 'Tau Protein', 'relation': 'potential adaptor', 'score': 0.65}, {'source': 'ARIH2', 'target': 'Tau Protein', 'relation': 'initiates monoubiquitination', 'score': 0.68}, {'source': 'RNF7', 'target': 'CUL5', 'relation': 'stabilizes complex', 'score': 0.85}, {'source': 'CHIP', 'target': 'Tau Protein', 'relation': 'ubiquitinates', 'score': 0.8}, {'source': 'CUL5', 'target': 'CRL5SOCS4 Complex', 'relation': 'forms part of', 'score': 0.95}, {'source': 'SOCS4', 'target': 'CRL5SOCS4 Complex', 'relation': 'component of', 'score': 0.9}, {'source': 'ELOB', 'target': 'CRL5SOCS4 Complex', 'relation': 'scaffold component', 'score': 0.95}, {'source': 'ELOC', 'target': 'CRL5SOCS4 Complex', 'relation': 'scaffold component', 'score': 0.95}, {'source': 'UFC1', 'target': 'UFM1', 'relation': 'conjugates', 'score': 0.87}, {'source': 'UFL1', 'target': 'UFM1', 'relation': 'ligase activity', 'score': 0.8}, {'source': 'UBA5', 'target': 'UFM1', 'relation': 'activates', 'score': 0.78}, {'source': 'UBE2F', 'target': 'CUL5', 'relation': 'mediates neddylation', 'score': 0.8}, {'source': 'PSME1', 'target': 'PA28', 'relation': 'forms subunit', 'score': 0.75}, {'source': 'PSME2', 'target': 'PA28', 'relation': 'forms subunit', 'score': 0.75}, {'source': 'MTOR', 'target': 'Autophagy', 'relation': 'negatively regulates', 'score': 0.9}, {'source': 'Autophagy', 'target': 'Tau Protein', 'relation': 'clears', 'score': 0.8}, {'source': 'Ubiquitin/Proteasome System', 'target': 'Tau Protein', 'relation': 'degrades', 'score': 0.85}, {'source': 'Oxidative Phosphorylation', 'target': 'Tau Oligomers', 'relation': 'inhibition increases', 'score': 0.85}, {'source': 'Electron Transport Chain', 'target': 'ROS Response', 'relation': 'generates', 'score': 0.9}, {'source': 'ROS Response', 'target': 'Tau 25kD Fragment', 'relation': 'induces formation', 'score': 0.85}, {'source': 'Neddylation', 'target': 'CRL5SOCS4 Complex', 'relation': 'activates', 'score': 0.85}, {'source': 'MAPT', 'target': 'Frontotemporal Dementia (FTD)', 'relation': 'mutations cause', 'score': 0.88}, {'source': 'Tau Protein', 'target': "Alzheimer's Disease", 'relation': 'aggregates in', 'score': 0.92}, {'source': 'KEAP1', 'target': 'ROS Response', 'relation': 'regulates oxidative stress', 'score': 0.65}, {'source': 'FECH', 'target': 'Electron Transport Chain', 'relation': 'supports mitochondrial function', 'score': 0.6}, {'source': 'FH', 'target': 'Electron Transport Chain', 'relation': 'supports mitochondrial function', 'score': 0.6}, {'source': 'WIPI2', 'target': 'Autophagy', 'relation': 'facilitates', 'score': 0.7}, {'source': 'CTSD', 'target': 'Autophagy', 'relation': 'mediates lysosomal degradation', 'score': 0.7}, {'source': 'PSAP', 'target': 'Autophagy', 'relation': 'supports lysosomal function', 'score': 0.7}, {'source': 'TSC1', 'target': 'Tau Protein', 'relation': 'negatively regulates', 'score': 0.9}, {'source': 'TSC1', 'target': 'mTOR Signaling', 'relation': 'component of', 'score': 0.9}, {'source': 'Chromatin Modification', 'target': 'Tau Protein', 'relation': 'modulates expression', 'score': 0.8}, {'source': 'Tau Protein', 'target': 'Progressive Supranuclear Palsy', 'relation': 'aggregates in', 'score': 0.88}, {'source': 'Tau Protein', 'target': 'Corticobasal Degeneration', 'relation': 'aggregates in', 'score': 0.88}, {'source': 'Tau Protein', 'target': 'Pick�s Disease', 'relation': 'aggregates in', 'score': 0.88}, {'source': 'MAPT', 'target': 'FTDP-17', 'relation': 'mutations cause', 'score': 0.88}, {'source': 'gRNA MAPT-1', 'target': 'MAPT', 'relation': 'targets', 'score': 0.85}, {'source': 'gRNA MAPT-2', 'target': 'MAPT', 'relation': 'targets', 'score': 0.75}, {'source': 'gRNA MAPT-1', 'target': 'Tau Protein', 'relation': 'reduces levels', 'score': 0.85}, {'source': 'gRNA MAPT-2', 'target': 'Tau Protein', 'relation': 'reduces levels', 'score': 0.75}, {'source': 'SH-SY5Y cell line', 'target': 'Tau Protein', 'relation': 'expresses', 'score': 0.9}, {'source': 'SH-SY5Y cell line', 'target': 'Cas9', 'relation': 'stably expresses', 'score': 0.95}, {'source': 'PC1', 'target': 'SH-SY5Y cell line', 'relation': 'derived from', 'score': 1.0}, {'source': 'SC2', 'target': 'PC1', 'relation': 'clone of', 'score': 1.0}, {'source': 'SP70 antibody', 'target': 'Tau Protein', 'relation': 'detects', 'score': 0.9}, {'source': 'SP70 antibody', 'target': 'FACS sorted low tau population', 'relation': 'identifies', 'score': 0.9}, {'source': 'gRNA MAPT-1', 'target': 'FACS sorted low tau population', 'relation': 'enriches', 'score': 0.85}, {'source': 'gRNA MAPT-2', 'target': 'FACS sorted low tau population', 'relation': 'enriches', 'score': 0.75}, {'source': 'Cas9', 'target': 'gRNA MAPT-1', 'relation': 'mediates editing with', 'score': 0.9}, {'source': 'Cas9', 'target': 'gRNA MAPT-2', 'relation': 'mediates editing with', 'score': 0.9}, {'source': 'MAPT', 'target': 'Tau Protein', 'relation': 'encodes', 'score': 0.99}, {'source': 'CUL5', 'target': 'Tau Protein', 'relation': 'ubiquitinates', 'score': 0.75}, {'source': 'SOCS4', 'target': 'Tau Protein', 'relation': 'recruits for ubiquitination', 'score': 0.7}, {'source': 'SOCS5', 'target': 'Tau Protein', 'relation': 'potential adaptor', 'score': 0.65}, {'source': 'ARIH2', 'target': 'Tau Protein', 'relation': 'initiates monoubiquitination', 'score': 0.68}, {'source': 'RNF7', 'target': 'CUL5', 'relation': 'stabilizes complex', 'score': 0.85}, {'source': 'CHIP', 'target': 'Tau Protein', 'relation': 'ubiquitinates', 'score': 0.8}, {'source': 'CUL5', 'target': 'CRL5SOCS4 Complex', 'relation': 'forms part of', 'score': 0.95}, {'source': 'SOCS4', 'target': 'CRL5SOCS4 Complex', 'relation': 'component of', 'score': 0.9}, {'source': 'ELOB', 'target': 'CRL5SOCS4 Complex', 'relation': 'scaffold component', 'score': 0.95}, {'source': 'ELOC', 'target': 'CRL5SOCS4 Complex', 'relation': 'scaffold component', 'score': 0.95}, {'source': 'UFC1', 'target': 'UFM1', 'relation': 'conjugates', 'score': 0.87}, {'source': 'UFL1', 'target': 'UFM1', 'relation': 'ligase activity', 'score': 0.8}, {'source': 'UBA5', 'target': 'UFM1', 'relation': 'activates', 'score': 0.78}, {'source': 'UBE2F', 'target': 'CUL5', 'relation': 'mediates neddylation', 'score': 0.8}, {'source': 'PSME1', 'target': 'PA28', 'relation': 'forms subunit', 'score': 0.75}, {'source': 'PSME2', 'target': 'PA28', 'relation': 'forms subunit', 'score': 0.75}, {'source': 'MTOR', 'target': 'Autophagy', 'relation': 'negatively regulates', 'score': 0.9}, {'source': 'Autophagy', 'target': 'Tau Protein', 'relation': 'clears', 'score': 0.8}, {'source': 'Ubiquitin/Proteasome System', 'target': 'Tau Protein', 'relation': 'degrades', 'score': 0.85}, {'source': 'Oxidative Phosphorylation', 'target': 'Tau Oligomers', 'relation': 'inhibition increases', 'score': 0.85}, {'source': 'Electron Transport Chain', 'target': 'ROS Response', 'relation': 'generates', 'score': 0.9}, {'source': 'ROS Response', 'target': 'Tau 25kD Fragment', 'relation': 'induces formation', 'score': 0.85}, {'source': 'Neddylation', 'target': 'CRL5SOCS4 Complex', 'relation': 'activates', 'score': 0.85}, {'source': 'MAPT', 'target': 'Frontotemporal Dementia (FTD)', 'relation': 'mutations cause', 'score': 0.88}, {'source': 'Tau Protein', 'target': "Alzheimer's Disease", 'relation': 'aggregates in', 'score': 0.92}, {'source': 'KEAP1', 'target': 'ROS Response', 'relation': 'regulates oxidative stress', 'score': 0.65}, {'source': 'FECH', 'target': 'Electron Transport Chain', 'relation': 'supports mitochondrial function', 'score': 0.6}, {'source': 'FH', 'target': 'Electron Transport Chain', 'relation': 'supports mitochondrial function', 'score': 0.6}, {'source': 'WIPI2', 'target': 'Autophagy', 'relation': 'facilitates', 'score': 0.7}, {'source': 'CTSD', 'target': 'Autophagy', 'relation': 'mediates lysosomal degradation', 'score': 0.7}, {'source': 'PSAP', 'target': 'Autophagy', 'relation': 'supports lysosomal function', 'score': 0.7}, {'source': 'UFMylation', 'target': 'Tau Seeding', 'relation': 'strongly modulates', 'score': 0.9}, {'source': 'Nuclear Mitochondrial Genes', 'target': 'Tau Seeding', 'relation': 'negatively modulates', 'score': 0.8}, {'source': 'CRL5SOCS4 Complex', 'target': 'Tau Protein', 'relation': 'controls soma-specific degradation', 'score': 0.8}, {'source': 'CUL5', 'target': "Alzheimer's Disease", 'relation': 'associated with neuronal resilience', 'score': 0.7}, {'source': 'CUL5', 'target': 'Neuro-immune Axis', 'relation': 'may modulate', 'score': 0.65}, {'source': 'Calpain', 'target': 'N-224 Tau Fragment', 'relation': 'produces', 'score': 0.9}, {'source': 'N-224 Tau Fragment', 'target': "Alzheimer's Disease", 'relation': 'serves as biomarker', 'score': 0.85}, {'source': 'ROS Response', 'target': 'Tau Protein', 'relation': 'oxidizes', 'score': 0.7}, {'source': 'MAPT', 'target': 'Tau Protein', 'relation': 'encodes', 'score': 0.99}, {'source': 'CUL5', 'target': 'Tau Protein', 'relation': 'ubiquitinates', 'score': 0.75}, {'source': 'SOCS4', 'target': 'Tau Protein', 'relation': 'recruits for ubiquitination', 'score': 0.7}, {'source': 'SOCS5', 'target': 'Tau Protein', 'relation': 'potential adaptor', 'score': 0.65}, {'source': 'ARIH2', 'target': 'Tau Protein', 'relation': 'initiates monoubiquitination', 'score': 0.68}, {'source': 'RNF7', 'target': 'CUL5', 'relation': 'stabilizes complex', 'score': 0.85}, {'source': 'CHIP', 'target': 'Tau Protein', 'relation': 'ubiquitinates', 'score': 0.8}, {'source': 'CUL5', 'target': 'CRL5SOCS4 Complex', 'relation': 'forms part of', 'score': 0.95}, {'source': 'SOCS4', 'target': 'CRL5SOCS4 Complex', 'relation': 'component of', 'score': 0.9}, {'source': 'ELOB', 'target': 'CRL5SOCS4 Complex', 'relation': 'scaffold component', 'score': 0.95}, {'source': 'ELOC', 'target': 'CRL5SOCS4 Complex', 'relation': 'scaffold component', 'score': 0.95}, {'source': 'UFC1', 'target': 'UFM1', 'relation': 'conjugates', 'score': 0.87}, {'source': 'UFL1', 'target': 'UFM1', 'relation': 'ligase activity', 'score': 0.8}, {'source': 'UBA5', 'target': 'UFM1', 'relation': 'activates', 'score': 0.78}, {'source': 'UBE2F', 'target': 'CUL5', 'relation': 'mediates neddylation', 'score': 0.8}, {'source': 'PSME1', 'target': 'PA28', 'relation': 'forms subunit', 'score': 0.75}, {'source': 'PSME2', 'target': 'PA28', 'relation': 'forms subunit', 'score': 0.75}, {'source': 'MTOR', 'target': 'Autophagy', 'relation': 'negatively regulates', 'score': 0.9}, {'source': 'Autophagy', 'target': 'Tau Protein', 'relation': 'clears', 'score': 0.8}, {'source': 'Ubiquitin/Proteasome System', 'target': 'Tau Protein', 'relation': 'degrades', 'score': 0.85}, {'source': 'Oxidative Phosphorylation', 'target': 'Tau Oligomers', 'relation': 'inhibition increases', 'score': 0.85}, {'source': 'Electron Transport Chain', 'target': 'ROS Response', 'relation': 'generates', 'score': 0.9}, {'source': 'ROS Response', 'target': 'Tau 25kD Fragment', 'relation': 'induces formation', 'score': 0.85}, {'source': 'Neddylation', 'target': 'CRL5SOCS4 Complex', 'relation': 'activates', 'score': 0.85}, {'source': 'MAPT', 'target': 'Frontotemporal Dementia (FTD)', 'relation': 'mutations cause', 'score': 0.88}, {'source': 'Tau Protein', 'target': "Alzheimer's Disease", 'relation': 'aggregates in', 'score': 0.92}, {'source': 'KEAP1', 'target': 'ROS Response', 'relation': 'regulates oxidative stress', 'score': 0.65}, {'source': 'FECH', 'target': 'Electron Transport Chain', 'relation': 'supports mitochondrial function', 'score': 0.6}, {'source': 'FH', 'target': 'Electron Transport Chain', 'relation': 'supports mitochondrial function', 'score': 0.6}, {'source': 'WIPI2', 'target': 'Autophagy', 'relation': 'facilitates', 'score': 0.7}, {'source': 'CTSD', 'target': 'Autophagy', 'relation': 'mediates lysosomal degradation', 'score': 0.7}, {'source': 'PSAP', 'target': 'Autophagy', 'relation': 'supports lysosomal function', 'score': 0.7}, {'source': 'TSC1', 'target': 'Tau Protein', 'relation': 'negatively regulates', 'score': 0.9}, {'source': 'Chromatin Modification', 'target': 'Tau Protein', 'relation': 'modulates expression', 'score': 0.8}, {'source': 'Tau Protein', 'target': 'Progressive Supranuclear Palsy', 'relation': 'aggregates in', 'score': 0.88}, {'source': 'Tau Protein', 'target': 'Corticobasal Degeneration', 'relation': 'aggregates in', 'score': 0.88}, {'source': 'Tau Protein', 'target': 'Pick�s Disease', 'relation': 'aggregates in', 'score': 0.88}, {'source': 'MAPT', 'target': 'FTDP-17', 'relation': 'mutations cause', 'score': 0.88}, {'source': 'gRNA MAPT-1', 'target': 'MAPT', 'relation': 'targets', 'score': 0.85}, {'source': 'gRNA MAPT-2', 'target': 'MAPT', 'relation': 'targets', 'score': 0.75}, {'source': 'gRNA MAPT-1', 'target': 'Tau Protein', 'relation': 'reduces levels', 'score': 0.85}, {'source': 'gRNA MAPT-2', 'target': 'Tau Protein', 'relation': 'reduces levels', 'score': 0.75}, {'source': 'SH-SY5Y cell line', 'target': 'Tau Protein', 'relation': 'expresses', 'score': 0.9}, {'source': 'SH-SY5Y cell line', 'target': 'Cas9', 'relation': 'stably expresses', 'score': 0.95}, {'source': 'PC1', 'target': 'SH-SY5Y cell line', 'relation': 'derived from', 'score': 1.0}, {'source': 'SC2', 'target': 'PC1', 'relation': 'clone of', 'score': 1.0}, {'source': 'SP70 antibody', 'target': 'Tau Protein', 'relation': 'detects', 'score': 0.9}, {'source': 'SP70 antibody', 'target': 'FACS sorted low tau population', 'relation': 'identifies', 'score': 0.9}, {'source': 'gRNA MAPT-1', 'target': 'FACS sorted low tau population', 'relation': 'enriches', 'score': 0.85}, {'source': 'gRNA MAPT-2', 'target': 'FACS sorted low tau population', 'relation': 'enriches', 'score': 0.75}, {'source': 'Cas9', 'target': 'gRNA MAPT-1', 'relation': 'mediates editing with', 'score': 0.9}, {'source': 'Cas9', 'target': 'gRNA MAPT-2', 'relation': 'mediates editing with', 'score': 0.9}, {'source': 'UFMylation', 'target': 'Tau Seeding', 'relation': 'strongly modulates', 'score': 0.9}, {'source': 'Nuclear Mitochondrial Genes', 'target': 'Tau Seeding', 'relation': 'negatively modulates', 'score': 0.8}, {'source': 'CRL5SOCS4 Complex', 'target': 'Tau Protein', 'relation': 'controls soma-specific degradation', 'score': 0.8}, {'source': 'CUL5', 'target': "Alzheimer's Disease", 'relation': 'associated with neuronal resilience', 'score': 0.7}, {'source': 'CUL5', 'target': 'Neuro-immune Axis', 'relation': 'may modulate', 'score': 0.65}, {'source': 'Calpain', 'target': 'N-224 Tau Fragment', 'relation': 'produces', 'score': 0.9}, {'source': 'N-224 Tau Fragment', 'target': "Alzheimer's Disease", 'relation': 'serves as biomarker', 'score': 0.85}, {'source': 'ROS Response', 'target': 'Tau Protein', 'relation': 'oxidizes', 'score': 0.7}, {'source': 'EIF4G2', 'target': 'Tau Protein', 'relation': 'positively regulates', 'score': 0.81}, {'source': 'MYT1', 'target': 'Tau Protein', 'relation': 'positively regulates', 'score': 0.79}, {'source': 'NSD1', 'target': 'Tau Protein', 'relation': 'positively regulates', 'score': 0.85}, {'source': 'NSD1', 'target': 'Chromatin Modification', 'relation': 'modulates', 'score': 0.85}, {'source': 'FBXW7', 'target': 'Tau Protein', 'relation': 'promotes degradation', 'score': 0.83}, {'source': 'FBXW7', 'target': 'CUL5', 'relation': 'functionally interacts', 'score': 0.8}, {'source': 'NCOA6', 'target': 'Tau Protein', 'relation': 'negatively regulates', 'score': 0.8}, {'source': 'TSC2', 'target': 'Tau Protein', 'relation': 'negatively regulates', 'score': 0.88}, {'source': 'TSC1', 'target': 'TSC2', 'relation': 'forms complex with', 'score': 0.95}, {'source': 'TSC2', 'target': 'TSC1', 'relation': 'forms complex with', 'score': 0.95}, {'source': 'TSC1', 'target': 'RHEB', 'relation': 'inhibits', 'score': 0.9}, {'source': 'TSC2', 'target': 'RHEB', 'relation': 'inhibits', 'score': 0.9}, {'source': 'RHEB', 'target': 'MTOR', 'relation': 'activates', 'score': 0.9}, {'source': 'MTOR', 'target': 'pS6', 'relation': 'phosphorylates', 'score': 0.9}, {'source': 'TSC2', 'target': 'Tau Oligomers', 'relation': 'modulates aggregation', 'score': 0.6}, {'source': 'NSD1', 'target': 'NCOA6', 'relation': 'co-regulates', 'score': 0.75}, {'source': 'SH-SY5Y cell line', 'target': 'PC1', 'relation': 'parent of', 'score': 0.95}, {'source': 'SH-SY5Y cell line', 'target': 'SC2', 'relation': 'parent of', 'score': 0.95}, {'source': 'Cas9', 'target': 'MAPT', 'relation': 'edits', 'score': 0.9}, {'source': 'gRNA MAPT-1', 'target': 'MAPT', 'relation': 'targets', 'score': 0.85}, {'source': 'gRNA MAPT-2', 'target': 'MAPT', 'relation': 'targets', 'score': 0.85}, {'source': 'SP70 antibody', 'target': 'Tau Protein', 'relation': 'detects', 'score': 0.95}, {'source': 'PC1', 'target': 'FACS sorted low tau population', 'relation': 'generates', 'score': 0.8}, {'source': 'SC2', 'target': 'FACS sorted low tau population', 'relation': 'generates', 'score': 0.8}]


nodes_data=[
  {
      "id": "2N4R tau (P301L-EGFP)",
      "type": "protein",
      "cluster": "Tauopathy",
      "size": 1.5,
      "PMID": 33536571
    },
    {
      "id": "384-well arrayed CRISPR screen",
      "type": "method",
      "cluster": "Screening",
      "size": 0.69,
      "PMID": 33536571
    },
    {
      "id": "ACVR1",
      "type": "gene",
      "cluster": "Gene Fingerprinting",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "ACVR1C",
      "type": "gene",
      "cluster": "Gene Fingerprinting",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "ACVR2A",
      "type": "gene",
      "cluster": "Gene Fingerprinting",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "AMPK Signaling",
      "type": "pathway",
      "cluster": "Metabolic Regulation",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "APPswe/swe",
      "type": "gene",
      "cluster": "AD Neurodegeneration",
      "size": 1.61,
      "PMID": "38917806"
    },
    {
      "id": "ARAF",
      "type": "gene",
      "cluster": "Other signaling",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "ARIH2",
      "type": "gene",
      "cluster": "Ubiquitin/Proteasome",
      "size": 1.39,
      "PMID": 37398204
    },
    {
      "id": "Alzheimer's Disease",
      "type": "disease",
      "cluster": "Tauopathy",
      "size": 2.71,
      "PMID": 37398204
    },
    {
      "id": "Autophagy",
      "type": "process",
      "cluster": "Autophagy",
      "size": 2.5,
      "PMID": 33536571
    },
    {
      "id": "Axon Guidance",
      "type": "pathway",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "A\u03b242",
      "type": "protein",
      "cluster": "AD Neurodegeneration",
      "size": 1.61,
      "PMID": "38917806"
    },
    {
      "id": "BRD2",
      "type": "gene",
      "cluster": "Chromatin Modification",
      "size": 0.69,
      "PMID": 34127790
    },
    {
      "id": "BRSK1",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "CAB39",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 1.1,
      "PMID": "37398204"
    },
    {
      "id": "CARD11",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "CCL2",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.69,
      "PMID": "37398204"
    },
    {
      "id": "CCL27",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "CCL8",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "CD14",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "CD2",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "CD40",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "CD40LG",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "CDK9",
      "type": "gene",
      "cluster": "Other signaling",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "CDKN1A",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "CEP170B",
      "type": "gene",
      "cluster": "Additional Hit Genes",
      "size": 0.1,
      "PMID": "38917806"
    },
    {
      "id": "CHIP",
      "type": "protein",
      "cluster": "Ubiquitin/Proteasome",
      "size": 1.39,
      "PMID": 37398204
    },
    {
      "id": "CHUK",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "CREBBP",
      "type": "gene",
      "cluster": "Chromatin Modification",
      "size": 0.69,
      "PMID": 34127790
    },
    {
      "id": "CRL5SOCS4 Complex",
      "type": "protein complex",
      "cluster": "Ubiquitin/Proteasome",
      "size": 2.3,
      "PMID": 37398204
    },
    {
      "id": "CSNK2A1",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "CSNK2B",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "CTSD",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "CUL5",
      "type": "gene",
      "cluster": "Ubiquitin/Proteasome",
      "size": 2.71,
      "PMID": 37398204
    },
    {
      "id": "Calpain",
      "type": "protein",
      "cluster": "Protease",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "Canonical glycolysis",
      "type": "process",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 1.8,
      "PMID": 33536571
    },
    {
      "id": "Cas9",
      "type": "protein",
      "cluster": "CRISPR",
      "size": 1.1,
      "PMID": 33536571
    },
    {
      "id": "Cellular Aging",
      "type": "biological process",
      "cluster": "Aging Pathways",
      "size": 2.48,
      "PMID": "38917806"
    },
    {
      "id": "Chromatin Modification",
      "type": "pathway",
      "cluster": "Chromatin Modification",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "Complex I biogenesis",
      "type": "process",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "Corticobasal Degeneration",
      "type": "disease",
      "cluster": "Tauopathy",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "Cytokine Response",
      "type": "pathway",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "DGKQ",
      "type": "gene",
      "cluster": "Mitochondrial Morphology",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "DNAJB11",
      "type": "gene",
      "cluster": "Additional Hit Genes",
      "size": 0.1,
      "PMID": "38917806"
    },
    {
      "id": "EDA2R",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "EFNA4",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "EIF4G2",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 1.0,
      "PMID": 37398204
    },
    {
      "id": "ELOB",
      "type": "gene",
      "cluster": "Ubiquitin/Proteasome",
      "size": 1.1,
      "PMID": 37398204
    },
    {
      "id": "ELOC",
      "type": "gene",
      "cluster": "Ubiquitin/Proteasome",
      "size": 1.1,
      "PMID": 37398204
    },
    {
      "id": "EPHB2",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "EPHB3",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "Electron Transport Chain",
      "type": "pathway",
      "cluster": "Mitochondrial Function",
      "size": 1.39,
      "PMID": 37398204
    },
    {
      "id": "Elongated mitochondria",
      "type": "phenotype",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 1.8,
      "PMID": 33536571
    },
    {
      "id": "FACS sorted low tau population",
      "type": "cell population",
      "cluster": "Cell Line",
      "size": 0.69,
      "PMID": 33536571
    },
    {
      "id": "FAM76B",
      "type": "gene",
      "cluster": "Additional Hit Genes",
      "size": 0.1,
      "PMID": "38917806"
    },
    {
      "id": "FAS",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "FBXO11",
      "type": "gene",
      "cluster": "Ubiquitin/Proteasome",
      "size": 0.69,
      "PMID": 34127790
    },
    {
      "id": "FBXW7",
      "type": "gene",
      "cluster": "Ubiquitin/Proteasome",
      "size": 1.0,
      "PMID": 37398204
    },
    {
      "id": "FECH",
      "type": "gene",
      "cluster": "Mitochondrial Function",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "FH",
      "type": "gene",
      "cluster": "Mitochondrial Function",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "FIG4",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "FMR1",
      "type": "gene",
      "cluster": "Fragile X & Lysosome",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "FTDP-17",
      "type": "disease",
      "cluster": "Tauopathy",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "FUS",
      "type": "gene",
      "cluster": "RNA metabolism",
      "size": 0.69,
      "PMID": 34127790
    },
    {
      "id": "FXR2",
      "type": "gene",
      "cluster": "Fragile X & Lysosome",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "FYN",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 1.1,
      "PMID": "37398204"
    },
    {
      "id": "Fragmented mitochondria",
      "type": "phenotype",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 1.8,
      "PMID": 33536571
    },
    {
      "id": "Frontotemporal Dementia (FTD)",
      "type": "disease",
      "cluster": "Tauopathy",
      "size": 2.3,
      "PMID": 37398204
    },
    {
      "id": "GPI-anchor Biosynthesis",
      "type": "pathway",
      "cluster": "Protein Modification",
      "size": 1.1,
      "PMID": 37398204
    },
    {
      "id": "GSK3A",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "GSK3B",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "Golgi",
      "type": "organelle",
      "cluster": "Cellular Organelles",
      "size": 0.69,
      "PMID": 33536571
    },
    {
      "id": "HDAC4",
      "type": "gene",
      "cluster": "Other signaling",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "High content imaging (HCI)",
      "type": "method",
      "cluster": "Screening",
      "size": 0.69,
      "PMID": 33536571
    },
    {
      "id": "IFI44",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "IGFBP3",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "IKBKB",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "IKBKG",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "IL17D",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "IL32",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "ILK",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "IMPA1",
      "type": "gene",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "ING1",
      "type": "gene",
      "cluster": "Mitochondrial Morphology",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "INPP1",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "INPP5E",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "IRAK2",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "Inflammatory signaling",
      "type": "pathway",
      "cluster": "Inflammatory Signaling",
      "size": 1.0,
      "PMID": 33536571
    },
    {
      "id": "KDM6A",
      "type": "gene",
      "cluster": "Chromatin Modification",
      "size": 0.69,
      "PMID": 34127790
    },
    {
      "id": "KDM7A",
      "type": "gene",
      "cluster": "Other signaling",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "KEAP1",
      "type": "gene",
      "cluster": "Oxidative Stress",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "KMT2D",
      "type": "gene",
      "cluster": "Chromatin Modification",
      "size": 0.69,
      "PMID": 34127790
    },
    {
      "id": "LAMP1",
      "type": "protein",
      "cluster": "Lysosome",
      "size": 1.79,
      "PMID": "37398204"
    },
    {
      "id": "LC3",
      "type": "protein",
      "cluster": "Autophagy",
      "size": 2.2,
      "PMID": "37398204"
    },
    {
      "id": "LIMK1",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "LIMK2",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "LRRK2G2019S/G2019S",
      "type": "gene",
      "cluster": "PD Neurodegeneration",
      "size": 0.1,
      "PMID": "38917806"
    },
    {
      "id": "LTK",
      "type": "gene",
      "cluster": "Other signaling",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "LYN",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "Lysosome homeostasis",
      "type": "process",
      "cluster": "Lysosome homeostasis",
      "size": 1.2,
      "PMID": 33536571
    },
    {
      "id": "Lysosomes",
      "type": "organelle",
      "cluster": "Cellular Organelles",
      "size": 0.69,
      "PMID": 33536571
    },
    {
      "id": "MAP3K7",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "MAP3K7CL",
      "type": "gene",
      "cluster": "Other signaling",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "MAP3K9",
      "type": "gene",
      "cluster": "Other signaling",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "MAP4K1",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "MAP4K4",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "MAPK Signaling",
      "type": "pathway",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "MAPK4",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "MAPKAPK2",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "MAPKAPK5",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "MAPT",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 3.22,
      "PMID": 37398204
    },
    {
      "id": "MARK1",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "MARK2",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 1.1,
      "PMID": "37398204"
    },
    {
      "id": "MET",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "METTL14",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 1.1,
      "PMID": 37398204
    },
    {
      "id": "METTL3",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 1.1,
      "PMID": 37398204
    },
    {
      "id": "MFN1",
      "type": "gene",
      "cluster": "Mitochondrial Morphology",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "MFN2",
      "type": "gene",
      "cluster": "Mitochondrial Morphology",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "MLN4924",
      "type": "chemical",
      "cluster": "Neddylation & Aging",
      "size": 2.08,
      "PMID": "38917806"
    },
    {
      "id": "MTA3",
      "type": "gene",
      "cluster": "Other signaling",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "MTM1",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "MTMR14",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "MTOR",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 1.61,
      "PMID": 37398204
    },
    {
      "id": "MYT1",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 1.0,
      "PMID": 37398204
    },
    {
      "id": "Mitochondrial Morphology",
      "type": "biological process",
      "cluster": "Mitochondrial Morphology",
      "size": 0.69,
      "PMID": "37398204"
    },
    {
      "id": "Mitochondrial fatty acid beta-oxidation",
      "type": "process",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "Mitochondrial function",
      "type": "process",
      "cluster": "Mitochondrial",
      "size": 0.2,
      "PMID": 33536571
    },
    {
      "id": "Mitochondrial membrane potential",
      "type": "process",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "Mitochondrial morphology",
      "type": "process",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 2.0,
      "PMID": 33536571
    },
    {
      "id": "Mitochondrial translation",
      "type": "process",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "Mitochondrial transport",
      "type": "process",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "Multiplexed gRNAs",
      "type": "reagent",
      "cluster": "CRISPR",
      "size": 0.69,
      "PMID": 33536571
    },
    {
      "id": "N-224 Tau Fragment",
      "type": "protein fragment",
      "cluster": "Tauopathy",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "NAE1",
      "type": "gene",
      "cluster": "Neddylation & Aging",
      "size": 1.1,
      "PMID": "38917806"
    },
    {
      "id": "NCOA6",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 1.0,
      "PMID": 37398204
    },
    {
      "id": "NF-\u00ce\u00baB pathway",
      "type": "pathway",
      "cluster": "Tauopathy",
      "size": 1.0,
      "PMID": 33536571
    },
    {
      "id": "NF-\u03baB Pathway",
      "type": "pathway",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 1.1,
      "PMID": "37398204"
    },
    {
      "id": "NF-\u03baB pathway",
      "type": "pathway",
      "cluster": "Inflammatory Signaling",
      "size": 1.0,
      "PMID": 33536571
    },
    {
      "id": "NLRP3",
      "type": "gene",
      "cluster": "Other",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "NRP1",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "NSD1",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 1.0,
      "PMID": 37398204
    },
    {
      "id": "NUDT6",
      "type": "gene",
      "cluster": "Additional Hit Genes",
      "size": 0.1,
      "PMID": "38917806"
    },
    {
      "id": "Neddylation",
      "type": "pathway",
      "cluster": "Protein Modification",
      "size": 1.39,
      "PMID": 37398204
    },
    {
      "id": "Neddylation Pathway",
      "type": "pathway",
      "cluster": "Neddylation & Aging",
      "size": 2.3,
      "PMID": "38917806"
    },
    {
      "id": "Neuro-immune Axis",
      "type": "pathway",
      "cluster": "Immune Regulation",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "Neurodegeneration",
      "type": "disease",
      "cluster": "AD Neurodegeneration",
      "size": 1.1,
      "PMID": "38917806"
    },
    {
      "id": "Nuclear Mitochondrial Genes",
      "type": "gene group",
      "cluster": "Mitochondrial Function",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "OCRL",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "Oxidative Phosphorylation",
      "type": "pathway",
      "cluster": "Mitochondrial Function",
      "size": 1.61,
      "PMID": 37398204
    },
    {
      "id": "Oxidative phosphorylation",
      "type": "process",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "Oxygen Consumption Rate (OCR)",
      "type": "measurement",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "PA28",
      "type": "protein complex",
      "cluster": "Proteasome Activation",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "PAK4",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PC1",
      "type": "cell line clone",
      "cluster": "Cell Line",
      "size": 1.1,
      "PMID": 33536571
    },
    {
      "id": "PFKFB4",
      "type": "gene",
      "cluster": "Other signaling",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "PGK1",
      "type": "gene",
      "cluster": "Other signaling",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "PHF11",
      "type": "gene",
      "cluster": "Other signaling",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "PHOX2A",
      "type": "gene",
      "cluster": "Transcription Factor",
      "size": 0.69,
      "PMID": 34127790
    },
    {
      "id": "PIK3AP1",
      "type": "gene",
      "cluster": "Gene Fingerprinting",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PIK3C2B",
      "type": "gene",
      "cluster": "Autophagy & Lysosome",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PIK3C3",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PIK3CA",
      "type": "gene",
      "cluster": "Autophagy & Lysosome",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PIK3CD",
      "type": "gene",
      "cluster": "Autophagy & Lysosome",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PIK3CG",
      "type": "gene",
      "cluster": "Autophagy & Lysosome",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PIK3R4",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PKLR",
      "type": "gene",
      "cluster": "Other signaling",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "PLXNA1",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PLXNA3",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PLXNA4",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PLXNB2",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PLXNB3",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PLXNC1",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PPIP5K1",
      "type": "gene",
      "cluster": "Mitochondrial Morphology",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PPM1E",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PPM1N",
      "type": "gene",
      "cluster": "Mitochondrial Morphology",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PPP1CB",
      "type": "gene",
      "cluster": "Additional Hit Genes",
      "size": 0.1,
      "PMID": "38917806"
    },
    {
      "id": "PPP1R3A",
      "type": "gene",
      "cluster": "Gene Fingerprinting",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PPP1R3B",
      "type": "gene",
      "cluster": "Gene Fingerprinting",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PPP2R1B",
      "type": "gene",
      "cluster": "Gene Fingerprinting",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PPP2R5C",
      "type": "gene",
      "cluster": "Gene Fingerprinting",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PPPIAL4E",
      "type": "gene",
      "cluster": "Additional Hit Genes",
      "size": 0.1,
      "PMID": "38917806"
    },
    {
      "id": "PRKAA1",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.69,
      "PMID": "37398204"
    },
    {
      "id": "PRKACA",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PRKCG",
      "type": "gene",
      "cluster": "Gene Fingerprinting",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "PSAP",
      "type": "gene",
      "cluster": "Lysosomal Function",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "PSENM146V/M146V",
      "type": "gene",
      "cluster": "AD Neurodegeneration",
      "size": 0.1,
      "PMID": "38917806"
    },
    {
      "id": "PSME1",
      "type": "gene",
      "cluster": "Proteasome Activation",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "PSME2",
      "type": "gene",
      "cluster": "Proteasome Activation",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "PXK",
      "type": "gene",
      "cluster": "Gene Fingerprinting",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "Phosphatidylinositol-3-phosphate biosynthetic process",
      "type": "process",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "Pick\u2019s Disease",
      "type": "disease",
      "cluster": "Tauopathy",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "Progressive Supranuclear Palsy",
      "type": "disease",
      "cluster": "Tauopathy",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "Proteostasis",
      "type": "pathway",
      "cluster": "Aging Pathways",
      "size": 0.69,
      "PMID": "38917806"
    },
    {
      "id": "RAB7A",
      "type": "gene",
      "cluster": "Endolysosomal",
      "size": 0.69,
      "PMID": 34127790
    },
    {
      "id": "RHEB",
      "type": "gene",
      "cluster": "mTOR Pathway",
      "size": 0.8,
      "PMID": 37398204
    },
    {
      "id": "RIPK1",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "RNA Degradation",
      "type": "pathway",
      "cluster": "Gene Expression",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "RNA Transport",
      "type": "pathway",
      "cluster": "Gene Expression",
      "size": 1.1,
      "PMID": 37398204
    },
    {
      "id": "RNA-seq analysis",
      "type": "method",
      "cluster": "Transcriptomics",
      "size": 0.69,
      "PMID": 33536571
    },
    {
      "id": "RNF7",
      "type": "gene",
      "cluster": "Ubiquitin/Proteasome",
      "size": 1.39,
      "PMID": 37398204
    },
    {
      "id": "ROCK1",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "ROS Response",
      "type": "pathway",
      "cluster": "Oxidative Stress",
      "size": 1.79,
      "PMID": 37398204
    },
    {
      "id": "RYK",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "Respiratory electron transport chain",
      "type": "process",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "Respiratory function",
      "type": "phenotype",
      "cluster": "Mitochondrial Function",
      "size": 0.69,
      "PMID": 33536571
    },
    {
      "id": "SACM1L",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "SC2",
      "type": "cell line clone",
      "cluster": "Cell Line",
      "size": 1.1,
      "PMID": 33536571
    },
    {
      "id": "SENP8",
      "type": "gene",
      "cluster": "Neddylation",
      "size": 0.69,
      "PMID": 34127790
    },
    {
      "id": "SETDB2",
      "type": "gene",
      "cluster": "Other signaling",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "SGK1",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "SH-SY5Y cell line",
      "type": "cell line",
      "cluster": "Cell Line",
      "size": 1.61,
      "PMID": 33536571
    },
    {
      "id": "SIK3",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "SIRT2",
      "type": "gene",
      "cluster": "Lysosome",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "SIRT3",
      "type": "gene",
      "cluster": "Lysosome",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "SIRT5",
      "type": "gene",
      "cluster": "Lysosome",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "SLK",
      "type": "gene",
      "cluster": "Gene Fingerprinting",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "SMARCA2",
      "type": "gene",
      "cluster": "Other signaling",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "SNRK",
      "type": "gene",
      "cluster": "Gene Fingerprinting",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "SOCS4",
      "type": "gene",
      "cluster": "Ubiquitin/Proteasome",
      "size": 2.3,
      "PMID": 37398204
    },
    {
      "id": "SOCS5",
      "type": "gene",
      "cluster": "Ubiquitin/Proteasome",
      "size": 1.1,
      "PMID": 37398204
    },
    {
      "id": "SP70 antibody",
      "type": "reagent",
      "cluster": "Antibody",
      "size": 0.69,
      "PMID": 33536571
    },
    {
      "id": "SQSTM1",
      "type": "protein",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "SRC",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "SSH2",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "SSH3",
      "type": "gene",
      "cluster": "MAPK & Axon Guidance",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "STK11",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 0.3,
      "PMID": 33536571
    },
    {
      "id": "STRADA",
      "type": "gene",
      "cluster": "Tauopathy",
      "size": 0.69,
      "PMID": "37398204"
    },
    {
      "id": "Seahorse ATP Production Rate assay",
      "type": "assay",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "Seahorse Glycolytic Rate assay",
      "type": "assay",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "Seahorse Mito Stress Test",
      "type": "assay",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "Synthetic tau fibrils",
      "type": "reagent",
      "cluster": "Tauopathy",
      "size": 0.69,
      "PMID": 33536571
    },
    {
      "id": "TCA Cycle",
      "type": "pathway",
      "cluster": "Mitochondrial Function",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "TCA cycle",
      "type": "process",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "TFEB",
      "type": "protein",
      "cluster": "Autophagy & Lysosome",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "TLR4",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "TMRE labeling",
      "type": "assay",
      "cluster": "Mitochondrial Morphology & Bioenergetics",
      "size": 0.1,
      "PMID": 33536571
    },
    {
      "id": "TNFRSF1A",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "TNFRSF9",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "TPTE",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "TPTE2",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "TRIM24",
      "type": "gene",
      "cluster": "Mitochondrial Morphology",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "TRIM28",
      "type": "gene",
      "cluster": "Chromatin Modification",
      "size": 0.69,
      "PMID": 34127790
    },
    {
      "id": "TSC1",
      "type": "gene",
      "cluster": "mTOR Pathway",
      "size": 1.1,
      "PMID": 37398204
    },
    {
      "id": "TSC2",
      "type": "gene",
      "cluster": "mTOR Pathway",
      "size": 1.1,
      "PMID": 37398204
    },
    {
      "id": "Tau",
      "type": "protein",
      "cluster": "AD Neurodegeneration",
      "size": 2.3,
      "PMID": "38917806"
    },
    {
      "id": "Tau 25kD Fragment",
      "type": "protein fragment",
      "cluster": "Tauopathy",
      "size": 2.08,
      "PMID": 37398204
    },
    {
      "id": "Tau Aggregation",
      "type": "biological process",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 1.61,
      "PMID": "37398204"
    },
    {
      "id": "Tau Oligomers",
      "type": "protein",
      "cluster": "Tauopathy",
      "size": 2.71,
      "PMID": 37398204
    },
    {
      "id": "Tau Protein",
      "type": "protein",
      "cluster": "Tauopathy",
      "size": 3.4,
      "PMID": 37398204
    },
    {
      "id": "Tau Seeding",
      "type": "process",
      "cluster": "Tauopathy",
      "size": 1.1,
      "PMID": 37398204
    },
    {
      "id": "UBA3",
      "type": "gene",
      "cluster": "Neddylation & Aging",
      "size": 1.61,
      "PMID": "38917806"
    },
    {
      "id": "UBA5",
      "type": "gene",
      "cluster": "UFMylation",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "UBE2A",
      "type": "gene",
      "cluster": "Ubiquitination",
      "size": 0.69,
      "PMID": 34127790
    },
    {
      "id": "UBE2F",
      "type": "gene",
      "cluster": "Neddylation",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "UBE2H",
      "type": "gene",
      "cluster": "Ubiquitination",
      "size": 0.69,
      "PMID": 34127790
    },
    {
      "id": "UBE4B",
      "type": "gene",
      "cluster": "Ubiquitination",
      "size": 0.69,
      "PMID": 34127790
    },
    {
      "id": "UFC1",
      "type": "gene",
      "cluster": "UFMylation",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "UFL1",
      "type": "gene",
      "cluster": "UFMylation",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "UFM1",
      "type": "gene",
      "cluster": "UFMylation",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "UFMylation",
      "type": "pathway",
      "cluster": "Protein Modification",
      "size": 1.61,
      "PMID": 37398204
    },
    {
      "id": "Ubiquitin/Proteasome System",
      "type": "pathway",
      "cluster": "Protein Clearance",
      "size": 2.08,
      "PMID": 37398204
    },
    {
      "id": "VCAM1",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "VPS33A",
      "type": "gene",
      "cluster": "Endolysosomal",
      "size": 0.69,
      "PMID": 34127790
    },
    {
      "id": "VPS36",
      "type": "gene",
      "cluster": "Additional Hit Genes",
      "size": 0.1,
      "PMID": "38917806"
    },
    {
      "id": "WIPI2",
      "type": "gene",
      "cluster": "Autophagy",
      "size": 0.69,
      "PMID": 37398204
    },
    {
      "id": "ZAP70",
      "type": "gene",
      "cluster": "Tau Aggregation & Inflammation",
      "size": 0.1,
      "PMID": "37398204"
    },
    {
      "id": "gRNA MAPT-1",
      "type": "reagent",
      "cluster": "CRISPR",
      "size": 0.69,
      "PMID": 33536571
    },
    {
      "id": "gRNA MAPT-2",
      "type": "reagent",
      "cluster": "CRISPR",
      "size": 0.69,
      "PMID": 33536571
    },
    {
      "id": "mTOR Signaling",
      "type": "pathway",
      "cluster": "Autophagy",
      "size": 1.79,
      "PMID": 37398204
    },
    {
      "id": "p62/SQSTM1",
      "type": "protein",
      "cluster": "Tauopathy",
      "size": 1.1,
      "PMID": 33536571
    },
    {
      "id": "pS6",
      "type": "protein",
      "cluster": "mTOR Pathway",
      "size": 0.8,
      "PMID": 37398204
    },
    {
      "id": "pTau",
      "type": "protein",
      "cluster": "AD Neurodegeneration",
      "size": 2.08,
      "PMID": "38917806"
    },
    {
      "id": "tau aggregation",
      "type": "process",
      "cluster": "Tauopathy",
      "size": 2.5,
      "PMID": 33536571
    }
  ]
  edges_data = [
    {
      "source": "2N4R tau (P301L-EGFP)",
      "target": "Tau Aggregation",
      "relation": "forms aggregates",
      "score": 0.9,
      "PMID": 33536571
    },
    {
      "source": "2N4R tau (P301L-EGFP)",
      "target": "Tau Protein",
      "relation": "derived from",
      "score": 0.95,
      "PMID": 33536571
    },
    {
      "source": "384-well arrayed CRISPR screen",
      "target": "High content imaging (HCI)",
      "relation": "enables phenotype readout",
      "score": 0.9,
      "PMID": 33536571
    },
    {
      "source": "ARAF",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "ARIH2",
      "target": "Tau Protein",
      "relation": "initiates monoubiquitination",
      "score": 0.68
    },
    {
      "source": "Autophagy",
      "target": "LC3",
      "relation": "stress increases LC3 puncta",
      "score": 0.9
    },
    {
      "source": "Autophagy",
      "target": "Lysosome homeostasis",
      "relation": "is coupled with",
      "score": 0.85
    },
    {
      "source": "Autophagy",
      "target": "Tau Protein",
      "relation": "clears",
      "score": 0.8
    },
    {
      "source": "A\u03b242",
      "target": "Neurodegeneration",
      "relation": "promotes",
      "score": 0.85
    },
    {
      "source": "BRD2",
      "target": "Tau Protein",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "BRSK1",
      "target": "Tau Protein",
      "relation": "modulates",
      "score": 0.75
    },
    {
      "source": "BRSK1",
      "target": "tau aggregation",
      "relation": "knockout increases aggregation",
      "score": 0.95
    },
    {
      "source": "CAB39",
      "target": "STK11",
      "relation": "forms complex with",
      "score": 0.9
    },
    {
      "source": "CAB39",
      "target": "STRADA",
      "relation": "forms complex with",
      "score": 0.88
    },
    {
      "source": "CAB39",
      "target": "tau aggregation",
      "relation": "knockout increases aggregation",
      "score": 0.95
    },
    {
      "source": "CARD11",
      "target": "NF-\u03baB Pathway",
      "relation": "associates with",
      "score": 0.8
    },
    {
      "source": "CARD11",
      "target": "tau aggregation",
      "relation": "associated with NF-\u00ce\u00baB pathway reducing aggregation",
      "score": 0.9
    },
    {
      "source": "CARD11",
      "target": "tau aggregation",
      "relation": "NF-\u00ce\u00baB associated; reduction decreases aggregation",
      "score": 0.9
    },
    {
      "source": "CCL2",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "CCL2",
      "target": "tau aggregation",
      "relation": "CRISPR disruption reduces aggregation; RNA-seq shows induction by aggregates",
      "score": 0.9
    },
    {
      "source": "CCL2",
      "target": "tau aggregation",
      "relation": "CRISPR disruption reduces aggregation; aggregates induce expression",
      "score": 0.9
    },
    {
      "source": "CCL27",
      "target": "tau aggregation",
      "relation": "RNA-seq: aggregates induce expression",
      "score": 0.9
    },
    {
      "source": "CCL8",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "CCL8",
      "target": "tau aggregation",
      "relation": "CRISPR disruption reduces aggregation",
      "score": 0.9
    },
    {
      "source": "CD14",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "CD14",
      "target": "tau aggregation",
      "relation": "CRISPR disruption reduces aggregation",
      "score": 0.9
    },
    {
      "source": "CD2",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "CD2",
      "target": "tau aggregation",
      "relation": "CRISPR disruption reduces aggregation",
      "score": 0.9
    },
    {
      "source": "CD40",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "CD40",
      "target": "tau aggregation",
      "relation": "CRISPR disruption reduces aggregation",
      "score": 0.9
    },
    {
      "source": "CD40LG",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "CD40LG",
      "target": "tau aggregation",
      "relation": "CRISPR disruption reduces aggregation",
      "score": 0.9
    },
    {
      "source": "CDK9",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "CDKN1A",
      "target": "tau aggregation",
      "relation": "RNA-seq: aggregates induce expression",
      "score": 0.9
    },
    {
      "source": "CHIP",
      "target": "Tau Protein",
      "relation": "ubiquitinates",
      "score": 0.8
    },
    {
      "source": "CHUK",
      "target": "NF-\u03baB Pathway",
      "relation": "activates",
      "score": 0.9
    },
    {
      "source": "CHUK",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.9
    },
    {
      "source": "CHUK",
      "target": "tau aggregation",
      "relation": "IKK complex disruption reduces aggregation",
      "score": 0.95
    },
    {
      "source": "CREBBP",
      "target": "Tau Protein",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "CRL5SOCS4 Complex",
      "target": "Tau Protein",
      "relation": "controls soma-specific degradation",
      "score": 0.8
    },
    {
      "source": "CSN5i3",
      "target": "Neurodegeneration",
      "relation": "increases (via deneddylation inhibition)",
      "score": 0.8
    },
    {
      "source": "CSNK2A1",
      "target": "NF-\u03baB Pathway",
      "relation": "associates with",
      "score": 0.8
    },
    {
      "source": "CSNK2A1",
      "target": "tau aggregation",
      "relation": "associated with NF-\u00ce\u00baB pathway reducing aggregation",
      "score": 0.9
    },
    {
      "source": "CSNK2A1",
      "target": "tau aggregation",
      "relation": "NF-\u00ce\u00baB associated; reduction decreases aggregation",
      "score": 0.9
    },
    {
      "source": "CSNK2B",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "CSNK2B",
      "target": "tau aggregation",
      "relation": "CRISPR disruption reduces aggregation",
      "score": 0.9
    },
    {
      "source": "CTSD",
      "target": "Autophagy",
      "relation": "mediates lysosomal degradation",
      "score": 0.7
    },
    {
      "source": "CUL5",
      "target": "Alzheimer's Disease",
      "relation": "associated with neuronal resilience",
      "score": 0.7
    },
    {
      "source": "CUL5",
      "target": "CRL5SOCS4 Complex",
      "relation": "forms part of",
      "score": 0.95
    },
    {
      "source": "CUL5",
      "target": "Neuro-immune Axis",
      "relation": "may modulate",
      "score": 0.65
    },
    {
      "source": "CUL5",
      "target": "Tau Protein",
      "relation": "ubiquitinates",
      "score": 0.75
    },
    {
      "source": "Calpain",
      "target": "N-224 Tau Fragment",
      "relation": "produces",
      "score": 0.9
    },
    {
      "source": "Cas9",
      "target": "gRNA MAPT-1",
      "relation": "mediates editing with",
      "score": 0.9
    },
    {
      "source": "Cas9",
      "target": "gRNA MAPT-2",
      "relation": "mediates editing with",
      "score": 0.9
    },
    {
      "source": "Cellular Aging",
      "target": "Tau Aggregation",
      "relation": "promotes",
      "score": 0.85
    },
    {
      "source": "Chromatin Modification",
      "target": "Tau Protein",
      "relation": "modulates expression",
      "score": 0.8
    },
    {
      "source": "DGKQ",
      "target": "Mitochondrial Morphology",
      "relation": "disrupts",
      "score": 0.75
    },
    {
      "source": "DGKQ",
      "target": "Mitochondrial function",
      "relation": "disruption reduces mitochondrial volume",
      "score": 0.88
    },
    {
      "source": "DNAJB11",
      "target": "Tau Aggregation",
      "relation": "regulates proteostasis impacting",
      "score": 0.75
    },
    {
      "source": "EDA2R",
      "target": "tau aggregation",
      "relation": "RNA-seq: aggregates induce expression",
      "score": 0.9
    },
    {
      "source": "EIF4G2",
      "target": "Tau Protein",
      "relation": "positively regulates",
      "score": 0.81
    },
    {
      "source": "EIF4G2",
      "target": "Tau Protein",
      "relation": "reduces",
      "score": 0.85
    },
    {
      "source": "ELOB",
      "target": "CRL5SOCS4 Complex",
      "relation": "scaffold component",
      "score": 0.95
    },
    {
      "source": "ELOC",
      "target": "CRL5SOCS4 Complex",
      "relation": "scaffold component",
      "score": 0.95
    },
    {
      "source": "EPHB2",
      "target": "Axon Guidance",
      "relation": "involved in",
      "score": 0.8
    },
    {
      "source": "Electron Transport Chain",
      "target": "ROS Response",
      "relation": "generates",
      "score": 0.9
    },
    {
      "source": "Elongated mitochondria",
      "target": "Canonical glycolysis",
      "relation": "upregulates gene expression",
      "score": 0.95
    },
    {
      "source": "Elongated mitochondria",
      "target": "Oxygen Consumption Rate (OCR)",
      "relation": "increases",
      "score": 0.95
    },
    {
      "source": "Elongated mitochondria",
      "target": "Seahorse ATP Production Rate assay",
      "relation": "associated with increased glycolysis",
      "score": 0.95
    },
    {
      "source": "Elongated mitochondria",
      "target": "Seahorse Glycolytic Rate assay",
      "relation": "associated with increased glycolysis",
      "score": 0.95
    },
    {
      "source": "Elongated mitochondria",
      "target": "Seahorse Mito Stress Test",
      "relation": "measured by",
      "score": 0.95
    },
    {
      "source": "FAS",
      "target": "tau aggregation",
      "relation": "RNA-seq: aggregates induce expression",
      "score": 0.9
    },
    {
      "source": "FBXO11",
      "target": "Tau Protein",
      "relation": "reduces",
      "score": 0.85
    },
    {
      "source": "FBXW7",
      "target": "CUL5",
      "relation": "functionally interacts",
      "score": 0.8
    },
    {
      "source": "FBXW7",
      "target": "Tau Protein",
      "relation": "promotes degradation",
      "score": 0.83
    },
    {
      "source": "FBXW7",
      "target": "Tau Protein",
      "relation": "reduces",
      "score": 0.85,
      "context": "SH-SY5Y"
    },
    {
      "source": "FBXW7",
      "target": "Tau Protein",
      "relation": "increases",
      "score": 0.9,
      "context": "iNgn2"
    },
    {
      "source": "FECH",
      "target": "Electron Transport Chain",
      "relation": "supports mitochondrial function",
      "score": 0.6
    },
    {
      "source": "FH",
      "target": "Electron Transport Chain",
      "relation": "supports mitochondrial function",
      "score": 0.6
    },
    {
      "source": "FIG4",
      "target": "Lysosome homeostasis",
      "relation": "modulates",
      "score": 0.92
    },
    {
      "source": "FIG4",
      "target": "MTMR14",
      "relation": "functionally interacts with",
      "score": 0.75
    },
    {
      "source": "FMR1",
      "target": "Lysosome",
      "relation": "modulates",
      "score": 0.8
    },
    {
      "source": "FMR1",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "FUS",
      "target": "Tau Protein",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "FXR2",
      "target": "Lysosome",
      "relation": "modulates",
      "score": 0.8
    },
    {
      "source": "FXR2",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "FYN",
      "target": "Tau Aggregation",
      "relation": "decreases",
      "score": 0.85
    },
    {
      "source": "FYN",
      "target": "tau aggregation",
      "relation": "disruption decreases aggregation",
      "score": 0.9
    },
    {
      "source": "Fragmented mitochondria",
      "target": "Complex I biogenesis",
      "relation": "downregulates gene expression",
      "score": 0.9
    },
    {
      "source": "Fragmented mitochondria",
      "target": "Mitochondrial fatty acid beta-oxidation",
      "relation": "downregulates gene expression",
      "score": 0.9
    },
    {
      "source": "Fragmented mitochondria",
      "target": "Mitochondrial translation",
      "relation": "downregulates gene expression",
      "score": 0.9
    },
    {
      "source": "Fragmented mitochondria",
      "target": "Mitochondrial transport",
      "relation": "downregulates gene expression",
      "score": 0.9
    },
    {
      "source": "Fragmented mitochondria",
      "target": "Oxygen Consumption Rate (OCR)",
      "relation": "decreases",
      "score": 0.95
    },
    {
      "source": "Fragmented mitochondria",
      "target": "Respiratory electron transport chain",
      "relation": "downregulates gene expression",
      "score": 0.9
    },
    {
      "source": "Fragmented mitochondria",
      "target": "TCA cycle",
      "relation": "downregulates gene expression",
      "score": 0.9
    },
    {
      "source": "GSK3A",
      "target": "Tau Protein",
      "relation": "phosphorylates",
      "score": 0.89
    },
    {
      "source": "GSK3A",
      "target": "tau aggregation",
      "relation": "disruption disassembles aggregates",
      "score": 0.95
    },
    {
      "source": "GSK3B",
      "target": "Tau Protein",
      "relation": "phosphorylates",
      "score": 0.9
    },
    {
      "source": "GSK3B",
      "target": "tau aggregation",
      "relation": "disruption disassembles aggregates",
      "score": 0.95
    },
    {
      "source": "HDAC4",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "High content imaging (HCI)",
      "target": "Tau Aggregation",
      "relation": "quantifies aggregates",
      "score": 0.9,
      "PMID": 33536571
    },
    {
      "source": "IFI44",
      "target": "tau aggregation",
      "relation": "RNA-seq: aggregates induce expression",
      "score": 0.9
    },
    {
      "source": "IGFBP3",
      "target": "tau aggregation",
      "relation": "RNA-seq: aggregates induce expression",
      "score": 0.9
    },
    {
      "source": "IKBKB",
      "target": "NF-\u03baB Pathway",
      "relation": "activates",
      "score": 0.9
    },
    {
      "source": "IKBKB",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.9
    },
    {
      "source": "IKBKB",
      "target": "tau aggregation",
      "relation": "IKK complex disruption reduces aggregation",
      "score": 0.95
    },
    {
      "source": "IKBKG",
      "target": "NF-\u03baB Pathway",
      "relation": "activates",
      "score": 0.9
    },
    {
      "source": "IKBKG",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.9
    },
    {
      "source": "IKBKG",
      "target": "tau aggregation",
      "relation": "IKK complex disruption reduces aggregation",
      "score": 0.95
    },
    {
      "source": "IL17D",
      "target": "tau aggregation",
      "relation": "RNA-seq: aggregates induce expression",
      "score": 0.9
    },
    {
      "source": "IL32",
      "target": "tau aggregation",
      "relation": "RNA-seq: aggregates induce expression",
      "score": 0.9
    },
    {
      "source": "IMPA1",
      "target": "Canonical glycolysis",
      "relation": "upregulates gene expression",
      "score": 0.95
    },
    {
      "source": "ING1",
      "target": "Mitochondrial Morphology",
      "relation": "promotes elongation",
      "score": 0.75
    },
    {
      "source": "ING1",
      "target": "Mitochondrial function",
      "relation": "disruption leads to elongated mitochondria",
      "score": 0.88
    },
    {
      "source": "INPP5E",
      "target": "Lysosome homeostasis",
      "relation": "modulates",
      "score": 0.92
    },
    {
      "source": "IRAK2",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "IRAK2",
      "target": "tau aggregation",
      "relation": "CRISPR disruption reduces aggregation",
      "score": 0.9
    },
    {
      "source": "Inflammatory signaling",
      "target": "Tau Aggregation",
      "relation": "modulates",
      "score": 0.8,
      "PMID": 33536571
    },
    {
      "source": "Inflammatory signaling",
      "target": "tau aggregation",
      "relation": "modulates",
      "score": 0.87
    },
    {
      "source": "KDM6A",
      "target": "Tau Protein",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "KDM7A",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "KEAP1",
      "target": "ROS Response",
      "relation": "regulates oxidative stress",
      "score": 0.65
    },
    {
      "source": "KMT2D",
      "target": "Tau Protein",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "LAMP1",
      "target": "Lysosome homeostasis",
      "relation": "is marker of",
      "score": 0.9
    },
    {
      "source": "LC3",
      "target": "Autophagy",
      "relation": "marks autophagosomes",
      "score": 0.9,
      "PMID": 33536571
    },
    {
      "source": "LC3",
      "target": "Autophagy",
      "relation": "is marker of",
      "score": 0.9
    },
    {
      "source": "LTK",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "LYN",
      "target": "NF-\u03baB Pathway",
      "relation": "associates with",
      "score": 0.8
    },
    {
      "source": "LYN",
      "target": "tau aggregation",
      "relation": "associated with NF-\u00ce\u00baB pathway reducing aggregation",
      "score": 0.9
    },
    {
      "source": "LYN",
      "target": "tau aggregation",
      "relation": "NF-\u00ce\u00baB associated; reduction decreases aggregation",
      "score": 0.9
    },
    {
      "source": "Lysosomes",
      "target": "Autophagy",
      "relation": "execute degradation",
      "score": 0.85,
      "PMID": 33536571
    },
    {
      "source": "MAP3K7CL",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "MAP3K9",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "MAP4K1",
      "target": "MAPK Signaling",
      "relation": "activates",
      "score": 0.8
    },
    {
      "source": "MAPT",
      "target": "FTDP-17",
      "relation": "mutations cause",
      "score": 0.88
    },
    {
      "source": "MAPT",
      "target": "Frontotemporal Dementia (FTD)",
      "relation": "mutations cause",
      "score": 0.88
    },
    {
      "source": "MAPT",
      "target": "Tau Protein",
      "relation": "encodes",
      "score": 0.99
    },
    {
      "source": "MARK1",
      "target": "Tau Protein",
      "relation": "phosphorylates",
      "score": 0.84
    },
    {
      "source": "MARK1",
      "target": "tau aggregation",
      "relation": "knockout increases aggregation",
      "score": 0.95
    },
    {
      "source": "MARK2",
      "target": "Tau Protein",
      "relation": "phosphorylates",
      "score": 0.85
    },
    {
      "source": "MARK2",
      "target": "tau aggregation",
      "relation": "knockout increases aggregation; overexpression decreases aggregation",
      "score": 0.95
    },
    {
      "source": "MFN1",
      "target": "Mitochondrial Morphology",
      "relation": "maintains",
      "score": 0.8
    },
    {
      "source": "MFN1",
      "target": "Mitochondrial function",
      "relation": "maintains normal mitochondrial morphology",
      "score": 0.9
    },
    {
      "source": "MFN2",
      "target": "Mitochondrial Morphology",
      "relation": "maintains",
      "score": 0.8
    },
    {
      "source": "MFN2",
      "target": "Mitochondrial function",
      "relation": "maintains normal mitochondrial morphology",
      "score": 0.9
    },
    {
      "source": "MLN4924",
      "target": "Neddylation Pathway",
      "relation": "inhibits",
      "score": 0.9
    },
    {
      "source": "MLN4924",
      "target": "Tau Aggregation",
      "relation": "increases",
      "score": 0.9
    },
    {
      "source": "MTA3",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "MTMR14",
      "target": "INPP5E",
      "relation": "regulates",
      "score": 0.7
    },
    {
      "source": "MTMR14",
      "target": "Lysosome homeostasis",
      "relation": "modulates",
      "score": 0.92
    },
    {
      "source": "MTOR",
      "target": "Autophagy",
      "relation": "negatively regulates",
      "score": 0.9
    },
    {
      "source": "MTOR",
      "target": "Tau Aggregation",
      "relation": "no significant effect",
      "score": 0.5
    },
    {
      "source": "MTOR",
      "target": "pS6",
      "relation": "phosphorylates",
      "score": 0.9
    },
    {
      "source": "MYT1",
      "target": "Tau Protein",
      "relation": "positively regulates",
      "score": 0.79
    },
    {
      "source": "MYT1",
      "target": "Tau Protein",
      "relation": "reduces",
      "score": 0.85
    },
    {
      "source": "Mitochondrial function",
      "target": "Respiratory function",
      "relation": "correlates with",
      "score": 0.85
    },
    {
      "source": "Mitochondrial morphology",
      "target": "Canonical glycolysis",
      "relation": "correlates with",
      "score": 0.9
    },
    {
      "source": "Mitochondrial morphology",
      "target": "Respiratory function",
      "relation": "correlates with",
      "score": 0.9
    },
    {
      "source": "Multiplexed gRNAs",
      "target": "384-well arrayed CRISPR screen",
      "relation": "enhances editing efficiency",
      "score": 0.85,
      "PMID": 33536571
    },
    {
      "source": "N-224 Tau Fragment",
      "target": "Alzheimer's Disease",
      "relation": "serves as biomarker",
      "score": 0.85
    },
    {
      "source": "NAE1",
      "target": "Neddylation Pathway",
      "relation": "regulatory subunit; essential for activity",
      "score": 0.9
    },
    {
      "source": "NCOA6",
      "target": "Tau Protein",
      "relation": "negatively regulates",
      "score": 0.8
    },
    {
      "source": "NCOA6",
      "target": "Tau Protein",
      "relation": "increases",
      "score": 0.9
    },
    {
      "source": "NF-\u03baB pathway",
      "target": "Tau Aggregation",
      "relation": "modulates",
      "score": 0.8,
      "PMID": 33536571
    },
    {
      "source": "NF-\u03baB pathway",
      "target": "tau aggregation",
      "relation": "modulates",
      "score": 0.88
    },
    {
      "source": "NLRP3",
      "target": "NF-\u03baB Pathway",
      "relation": "activates",
      "score": 0.85
    },
    {
      "source": "NSD1",
      "target": "NCOA6",
      "relation": "co-regulates",
      "score": 0.75
    },
    {
      "source": "NSD1",
      "target": "Tau Protein",
      "relation": "positively regulates",
      "score": 0.85
    },
    {
      "source": "NSD1",
      "target": "Tau Protein",
      "relation": "reduces",
      "score": 0.9
    },
    {
      "source": "NUDT6",
      "target": "Neurodegeneration",
      "relation": "regulates viability",
      "score": 0.7
    },
    {
      "source": "Neddylation",
      "target": "CRL5SOCS4 Complex",
      "relation": "activates",
      "score": 0.85
    },
    {
      "source": "Neddylation Pathway",
      "target": "Cellular Aging",
      "relation": "modulates; inhibition accelerates aging hallmarks",
      "score": 0.88
    },
    {
      "source": "Neddylation Pathway",
      "target": "Proteostasis",
      "relation": "impairs when inhibited",
      "score": 0.8
    },
    {
      "source": "Nuclear Mitochondrial Genes",
      "target": "Tau Seeding",
      "relation": "negatively modulates",
      "score": 0.8
    },
    {
      "source": "Oxidative Phosphorylation",
      "target": "Tau Oligomers",
      "relation": "inhibition increases",
      "score": 0.85
    },
    {
      "source": "PC1",
      "target": "SH-SY5Y cell line",
      "relation": "derived from",
      "score": 1.0
    },
    {
      "source": "PFKFB4",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "PGK1",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "PHF11",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "PHOX2A",
      "target": "Tau Protein",
      "relation": "increases",
      "score": 0.8
    },
    {
      "source": "PIK3AP1",
      "target": "PRKCG",
      "relation": "co-clusters with",
      "score": 0.75
    },
    {
      "source": "PIK3C2B",
      "target": "Tau Aggregation",
      "relation": "increases",
      "score": 0.85
    },
    {
      "source": "PIK3C2B",
      "target": "tau aggregation",
      "relation": "CRISPR perturbation increases aggregation",
      "score": 0.92
    },
    {
      "source": "PIK3C3",
      "target": "Autophagy",
      "relation": "initiates",
      "score": 0.96
    },
    {
      "source": "PIK3C3",
      "target": "Mitochondrial Morphology",
      "relation": "disrupts",
      "score": 0.75
    },
    {
      "source": "PIK3C3",
      "target": "PIK3R4",
      "relation": "forms complex with",
      "score": 0.87
    },
    {
      "source": "PIK3CA",
      "target": "Tau Aggregation",
      "relation": "increases",
      "score": 0.85
    },
    {
      "source": "PIK3CA",
      "target": "tau aggregation",
      "relation": "CRISPR perturbation increases aggregation",
      "score": 0.92
    },
    {
      "source": "PIK3CD",
      "target": "Tau Aggregation",
      "relation": "increases",
      "score": 0.85
    },
    {
      "source": "PIK3CD",
      "target": "tau aggregation",
      "relation": "CRISPR perturbation increases aggregation",
      "score": 0.92
    },
    {
      "source": "PIK3CG",
      "target": "Tau Aggregation",
      "relation": "increases",
      "score": 0.85
    },
    {
      "source": "PIK3CG",
      "target": "tau aggregation",
      "relation": "CRISPR perturbation increases aggregation",
      "score": 0.92
    },
    {
      "source": "PIK3R4",
      "target": "Autophagy",
      "relation": "initiates",
      "score": 0.96
    },
    {
      "source": "PKLR",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "PPIP5K1",
      "target": "Canonical glycolysis",
      "relation": "upregulates gene expression",
      "score": 0.95
    },
    {
      "source": "PPIP5K1",
      "target": "Mitochondrial Morphology",
      "relation": "promotes elongation",
      "score": 0.75
    },
    {
      "source": "PPIP5K1",
      "target": "Mitochondrial function",
      "relation": "disruption leads to elongated mitochondria",
      "score": 0.88
    },
    {
      "source": "PPM1E",
      "target": "Autophagy",
      "relation": "promotes",
      "score": 0.9
    },
    {
      "source": "PPM1N",
      "target": "Mitochondrial Morphology",
      "relation": "promotes elongation",
      "score": 0.75
    },
    {
      "source": "PPM1N",
      "target": "Mitochondrial function",
      "relation": "disruption leads to elongated mitochondria",
      "score": 0.88
    },
    {
      "source": "PPP1CB",
      "target": "Tau Aggregation",
      "relation": "modulates phosphorylation",
      "score": 0.75
    },
    {
      "source": "PPPIAL4E",
      "target": "Neurodegeneration",
      "relation": "regulates viability",
      "score": 0.7
    },
    {
      "source": "PRKAA1",
      "target": "Autophagy",
      "relation": "activates",
      "score": 0.95
    },
    {
      "source": "PRKACA",
      "target": "Autophagy",
      "relation": "promotes",
      "score": 0.9
    },
    {
      "source": "PSAP",
      "target": "Autophagy",
      "relation": "supports lysosomal function",
      "score": 0.7
    },
    {
      "source": "PSME1",
      "target": "PA28",
      "relation": "forms subunit",
      "score": 0.75
    },
    {
      "source": "PSME2",
      "target": "PA28",
      "relation": "forms subunit",
      "score": 0.75
    },
    {
      "source": "Phosphatidylinositol-3-phosphate biosynthetic process",
      "target": "Autophagy",
      "relation": "is enriched in",
      "score": 0.8
    },
    {
      "source": "Proteostasis",
      "target": "Cellular Aging",
      "relation": "disruption promotes",
      "score": 0.75
    },
    {
      "source": "RAB7A",
      "target": "Tau Protein",
      "relation": "increases",
      "score": 0.8
    },
    {
      "source": "RHEB",
      "target": "MTOR",
      "relation": "activates",
      "score": 0.9
    },
    {
      "source": "RIPK1",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "RIPK1",
      "target": "tau aggregation",
      "relation": "CRISPR disruption reduces aggregation",
      "score": 0.9
    },
    {
      "source": "RNA-seq analysis",
      "target": "Multiplexed gRNAs",
      "relation": "validates gene-specific effects",
      "score": 0.8,
      "PMID": 33536571
    },
    {
      "source": "RNF7",
      "target": "CUL5",
      "relation": "stabilizes complex",
      "score": 0.85
    },
    {
      "source": "ROS Response",
      "target": "Tau 25kD Fragment",
      "relation": "induces formation",
      "score": 0.85
    },
    {
      "source": "ROS Response",
      "target": "Tau Protein",
      "relation": "oxidizes",
      "score": 0.7
    },
    {
      "source": "SC2",
      "target": "PC1",
      "relation": "clone of",
      "score": 1.0
    },
    {
      "source": "SENP8",
      "target": "Neddylation Pathway",
      "relation": "regulates (deneddylation cycles)",
      "score": 0.8
    },
    {
      "source": "SENP8",
      "target": "Tau Protein",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "SETDB2",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "SGK1",
      "target": "Autophagy",
      "relation": "promotes",
      "score": 0.9
    },
    {
      "source": "SH-SY5Y cell line",
      "target": "Cas9",
      "relation": "stably expresses",
      "score": 0.95
    },
    {
      "source": "SH-SY5Y cell line",
      "target": "Tau Protein",
      "relation": "expresses",
      "score": 0.9
    },
    {
      "source": "SIK3",
      "target": "Tau Protein",
      "relation": "modulates",
      "score": 0.75
    },
    {
      "source": "SIK3",
      "target": "tau aggregation",
      "relation": "knockout increases aggregation",
      "score": 0.95
    },
    {
      "source": "SIRT2",
      "target": "Autophagy",
      "relation": "inhibits",
      "score": 0.93
    },
    {
      "source": "SIRT2",
      "target": "LAMP1",
      "relation": "regulates",
      "score": 0.8
    },
    {
      "source": "SIRT2",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "SIRT3",
      "target": "Autophagy",
      "relation": "inhibits",
      "score": 0.93
    },
    {
      "source": "SIRT3",
      "target": "LAMP1",
      "relation": "regulates",
      "score": 0.8
    },
    {
      "source": "SIRT3",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "SIRT5",
      "target": "Autophagy",
      "relation": "inhibits",
      "score": 0.93
    },
    {
      "source": "SIRT5",
      "target": "LAMP1",
      "relation": "regulates",
      "score": 0.8
    },
    {
      "source": "SIRT5",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "SMARCA2",
      "target": "tau aggregation",
      "relation": "loss increases aggregation",
      "score": 0.88
    },
    {
      "source": "SOCS4",
      "target": "CRL5SOCS4 Complex",
      "relation": "component of",
      "score": 0.9
    },
    {
      "source": "SOCS4",
      "target": "Tau Protein",
      "relation": "recruits for ubiquitination",
      "score": 0.7
    },
    {
      "source": "SOCS5",
      "target": "Tau Protein",
      "relation": "potential adaptor",
      "score": 0.65
    },
    {
      "source": "SP70 antibody",
      "target": "FACS sorted low tau population",
      "relation": "identifies",
      "score": 0.9
    },
    {
      "source": "SP70 antibody",
      "target": "Tau Protein",
      "relation": "detects",
      "score": 0.9
    },
    {
      "source": "STK11",
      "target": "MARK2",
      "relation": "phosphorylates",
      "score": 0.85
    },
    {
      "source": "STK11",
      "target": "NF-\u03baB pathway",
      "relation": "regulates inflammatory signaling",
      "score": 0.86
    },
    {
      "source": "STK11",
      "target": "STRADA",
      "relation": "forms complex with",
      "score": 0.88
    },
    {
      "source": "STK11",
      "target": "tau aggregation",
      "relation": "knockout increases aggregation; overexpression decreases aggregation",
      "score": 0.95
    },
    {
      "source": "STRADA",
      "target": "MARK2",
      "relation": "associates with",
      "score": 0.8
    },
    {
      "source": "STRADA",
      "target": "tau aggregation",
      "relation": "knockout increases aggregation",
      "score": 0.95
    },
    {
      "source": "Seahorse ATP Production Rate assay",
      "target": "Canonical glycolysis",
      "relation": "measures glycolysis",
      "score": 0.9
    },
    {
      "source": "Seahorse ATP Production Rate assay",
      "target": "Oxidative phosphorylation",
      "relation": "measures OXPHOS",
      "score": 0.9
    },
    {
      "source": "Seahorse Glycolytic Rate assay",
      "target": "Canonical glycolysis",
      "relation": "measures",
      "score": 0.95
    },
    {
      "source": "Seahorse Mito Stress Test",
      "target": "Oxygen Consumption Rate (OCR)",
      "relation": "measures",
      "score": 0.95
    },
    {
      "source": "Synthetic tau fibrils",
      "target": "2N4R tau (P301L-EGFP)",
      "relation": "induces aggregation",
      "score": 0.85,
      "PMID": 33536571
    },
    {
      "source": "TFEB",
      "target": "Lysosome",
      "relation": "regulates",
      "score": 0.85
    },
    {
      "source": "TLR4",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "TLR4",
      "target": "tau aggregation",
      "relation": "CRISPR disruption reduces aggregation",
      "score": 0.9
    },
    {
      "source": "TMRE labeling",
      "target": "Mitochondrial membrane potential",
      "relation": "measures",
      "score": 0.95
    },
    {
      "source": "TNFRSF1A",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "TNFRSF1A",
      "target": "tau aggregation",
      "relation": "CRISPR disruption reduces aggregation",
      "score": 0.9
    },
    {
      "source": "TNFRSF9",
      "target": "tau aggregation",
      "relation": "RNA-seq: aggregates induce expression",
      "score": 0.9
    },
    {
      "source": "TRIM24",
      "target": "Mitochondrial Morphology",
      "relation": "disrupts",
      "score": 0.75
    },
    {
      "source": "TRIM24",
      "target": "Mitochondrial function",
      "relation": "disruption reduces mitochondrial volume",
      "score": 0.88
    },
    {
      "source": "TRIM28",
      "target": "Tau Protein",
      "relation": "reduces",
      "score": 0.85
    },
    {
      "source": "TSC1",
      "target": "RHEB",
      "relation": "inhibits",
      "score": 0.9
    },
    {
      "source": "TSC1",
      "target": "TSC2",
      "relation": "forms complex with",
      "score": 0.95
    },
    {
      "source": "TSC1",
      "target": "Tau Aggregation",
      "relation": "no significant effect",
      "score": 0.5
    },
    {
      "source": "TSC1",
      "target": "Tau Protein",
      "relation": "negatively regulates",
      "score": 0.9
    },
    {
      "source": "TSC1",
      "target": "Tau Protein",
      "relation": "increases",
      "score": 0.9
    },
    {
      "source": "TSC1",
      "target": "mTOR Signaling",
      "relation": "component of",
      "score": 0.9
    },
    {
      "source": "TSC2",
      "target": "RHEB",
      "relation": "inhibits",
      "score": 0.9
    },
    {
      "source": "TSC2",
      "target": "TSC1",
      "relation": "forms complex with",
      "score": 0.95
    },
    {
      "source": "TSC2",
      "target": "Tau Aggregation",
      "relation": "slightly reduces",
      "score": 0.6
    },
    {
      "source": "TSC2",
      "target": "Tau Oligomers",
      "relation": "modulates aggregation",
      "score": 0.6
    },
    {
      "source": "TSC2",
      "target": "Tau Protein",
      "relation": "negatively regulates",
      "score": 0.88
    },
    {
      "source": "TSC2",
      "target": "Tau Protein",
      "relation": "increases",
      "score": 0.9
    },
    {
      "source": "Tau Aggregation",
      "target": "pTau",
      "relation": "correlates with",
      "score": 0.9
    },
    {
      "source": "Tau Protein",
      "target": "Alzheimer's Disease",
      "relation": "aggregates in",
      "score": 0.92
    },
    {
      "source": "Tau Protein",
      "target": "Corticobasal Degeneration",
      "relation": "aggregates in",
      "score": 0.88
    },
    {
      "source": "Tau Protein",
      "target": "Pick\u2019s Disease",
      "relation": "aggregates in",
      "score": 0.88
    },
    {
      "source": "Tau Protein",
      "target": "Progressive Supranuclear Palsy",
      "relation": "aggregates in",
      "score": 0.88
    },
    {
      "source": "UBA3",
      "target": "Neddylation Pathway",
      "relation": "catalytic subunit; loss impairs",
      "score": 0.9
    },
    {
      "source": "UBA3",
      "target": "Neurodegeneration",
      "relation": "synergizes with APPswe/swe to trigger",
      "score": 0.9
    },
    {
      "source": "UBA3",
      "target": "Tau Aggregation",
      "relation": "loss increases",
      "score": 0.85
    },
    {
      "source": "UBA5",
      "target": "UFM1",
      "relation": "activates",
      "score": 0.78
    },
    {
      "source": "UBE2A",
      "target": "Tau Protein",
      "relation": "increases",
      "score": 0.8
    },
    {
      "source": "UBE2F",
      "target": "CUL5",
      "relation": "mediates neddylation",
      "score": 0.8
    },
    {
      "source": "UBE2H",
      "target": "Tau Protein",
      "relation": "increases",
      "score": 0.8
    },
    {
      "source": "UBE4B",
      "target": "Tau Protein",
      "relation": "increases",
      "score": 0.8
    },
    {
      "source": "UFC1",
      "target": "UFM1",
      "relation": "conjugates",
      "score": 0.87
    },
    {
      "source": "UFL1",
      "target": "UFM1",
      "relation": "ligase activity",
      "score": 0.8
    },
    {
      "source": "UFMylation",
      "target": "Tau Seeding",
      "relation": "strongly modulates",
      "score": 0.9
    },
    {
      "source": "Ubiquitin/Proteasome System",
      "target": "Tau Protein",
      "relation": "degrades",
      "score": 0.85
    },
    {
      "source": "VCAM1",
      "target": "Tau Aggregation",
      "relation": "reduces",
      "score": 0.8
    },
    {
      "source": "VCAM1",
      "target": "tau aggregation",
      "relation": "CRISPR disruption reduces aggregation",
      "score": 0.9
    },
    {
      "source": "VPS33A",
      "target": "Tau Protein",
      "relation": "increases",
      "score": 0.8
    },
    {
      "source": "VPS36",
      "target": "Tau Aggregation",
      "relation": "modulates",
      "score": 0.75
    },
    {
      "source": "WIPI2",
      "target": "Autophagy",
      "relation": "facilitates",
      "score": 0.7
    },
    {
      "source": "ZAP70",
      "target": "NF-\u03baB Pathway",
      "relation": "associates with",
      "score": 0.8
    },
    {
      "source": "ZAP70",
      "target": "tau aggregation",
      "relation": "associated with NF-\u00ce\u00baB pathway reducing aggregation",
      "score": 0.9
    },
    {
      "source": "ZAP70",
      "target": "tau aggregation",
      "relation": "NF-\u00ce\u00baB associated; reduction decreases aggregation",
      "score": 0.9
    },
    {
      "source": "gRNA MAPT-1",
      "target": "FACS sorted low tau population",
      "relation": "enriches",
      "score": 0.85
    },
    {
      "source": "gRNA MAPT-1",
      "target": "MAPT",
      "relation": "targets",
      "score": 0.85
    },
    {
      "source": "gRNA MAPT-1",
      "target": "Tau Protein",
      "relation": "reduces levels",
      "score": 0.85
    },
    {
      "source": "gRNA MAPT-2",
      "target": "FACS sorted low tau population",
      "relation": "enriches",
      "score": 0.75
    },
    {
      "source": "gRNA MAPT-2",
      "target": "MAPT",
      "relation": "targets",
      "score": 0.75
    },
    {
      "source": "gRNA MAPT-2",
      "target": "Tau Protein",
      "relation": "reduces levels",
      "score": 0.75
    },
    {
      "source": "p62/SQSTM1",
      "target": "Tau Protein",
      "relation": "binds aggregated",
      "score": 0.9
    },
    {
      "source": "tau aggregation",
      "target": "Lysosome homeostasis",
      "relation": "aggregates alter lysosome dynamics",
      "score": 0.9
    }
  ]
  clusters_data = {
    "Tauopathy": [
      "2N4R tau (P301L-EGFP)",
      "Alzheimer's Disease",
      "BRSK1",
      "CAB39",
      "Corticobasal Degeneration",
      "EIF4G2",
      "FTDP-17",
      "FYN",
      "Frontotemporal Dementia (FTD)",
      "GSK3A",
      "GSK3B",
      "Inflammatory signaling",
      "MAPT",
      "MARK1",
      "MARK2",
      "METTL14",
      "METTL3",
      "MYT1",
      "N-224 Tau Fragment",
      "NCOA6",
      "NF-\u03baB pathway",
      "NSD1",
      "Pick\u2019s Disease",
      "Progressive Supranuclear Palsy",
      "SIK3",
      "STK11",
      "STRADA",
      "Synthetic tau fibrils",
      "Tau 25kD Fragment",
      "Tau Aggregation",
      "Tau Oligomers",
      "Tau Protein",
      "Tau Seeding",
      "p62/SQSTM1",
      "tau aggregation"
    ],
    "Autophagy": [
      "Autophagy",
      "CTSD",
      "FIG4",
      "INPP1",
      "INPP5E",
      "LC3",
      "MTM1",
      "MTMR14",
      "MTOR",
      "OCRL",
      "PIK3C2B",
      "PIK3C3",
      "PIK3CA",
      "PIK3CD",
      "PIK3CG",
      "PIK3R4",
      "PPM1E",
      "PRKAA1",
      "PRKACA",
      "Phosphatidylinositol-3-phosphate biosynthetic process",
      "SACM1L",
      "SGK1",
      "SIRT2",
      "SIRT3",
      "SIRT5",
      "SQSTM1",
      "TPTE",
      "TPTE2",
      "WIPI2",
      "mTOR Signaling"
    ],
    "Ubiquitin/Proteasome": [
      "ARIH2",
      "CHIP",
      "CRL5SOCS4 Complex",
      "CUL5",
      "ELOB",
      "ELOC",
      "FBXO11",
      "FBXW7",
      "RNF7",
      "SOCS4",
      "SOCS5"
    ],
    "UFMylation": [
      "UBA5",
      "UFC1",
      "UFL1",
      "UFM1"
    ],
    "Neddylation": [
      "SENP8",
      "UBE2F"
    ],
    "Proteasome Activation": [
      "PA28",
      "PSME1",
      "PSME2"
    ],
    "Oxidative Stress": [
      "KEAP1",
      "ROS Response"
    ],
    "Mitochondrial Function": [
      "Electron Transport Chain",
      "FECH",
      "FH",
      "Mitochondrial morphology",
      "Nuclear Mitochondrial Genes",
      "Oxidative Phosphorylation",
      "Respiratory function",
      "TCA Cycle"
    ],
    "Lysosomal Function": [
      "PSAP"
    ],
    "Protease": [
      "Calpain"
    ],
    "Protein Clearance": [
      "Autophagy",
      "Ubiquitin/Proteasome System"
    ],
    "Protein Modification": [
      "GPI-anchor Biosynthesis",
      "Neddylation",
      "UFMylation"
    ],
    "Gene Expression": [
      "RNA Degradation",
      "RNA Transport"
    ],
    "Metabolic Regulation": [
      "AMPK Signaling"
    ],
    "Immune Regulation": [
      "Neuro-immune Axis"
    ],
    "mTOR Pathway": [
      "RHEB",
      "TSC1",
      "TSC2",
      "pS6"
    ],
    "Chromatin Modification": [
      "BRD2",
      "CREBBP",
      "Chromatin Modification",
      "KDM6A",
      "KMT2D",
      "TRIM28"
    ],
    "RNA metabolism": [
      "FUS"
    ],
    "Transcription Factor": [
      "PHOX2A"
    ],
    "Endolysosomal": [
      "RAB7A",
      "VPS33A"
    ],
    "Ubiquitination": [
      "UBE2A",
      "UBE2H",
      "UBE4B"
    ],
    "Inflammatory Signaling": [
      "Inflammatory signaling",
      "NF-\u03baB pathway"
    ],
    "CRISPR": [
      "Cas9",
      "Multiplexed gRNAs",
      "gRNA MAPT-1",
      "gRNA MAPT-2"
    ],
    "Screening": [
      "384-well arrayed CRISPR screen",
      "High content imaging (HCI)"
    ],
    "Transcriptomics": [
      "RNA-seq analysis"
    ],
    "Cellular Organelles": [
      "Golgi",
      "Lysosomes"
    ],
    "Cell Line": [
      "FACS sorted low tau population",
      "PC1",
      "SC2",
      "SH-SY5Y cell line"
    ],
    "Antibody": [
      "SP70 antibody"
    ],
    "Lysosome": [
      "LAMP1",
      "SIRT2",
      "SIRT3",
      "SIRT5"
    ],
    "Tau Aggregation & Inflammation": [
      "CARD11",
      "CCL2",
      "CCL27",
      "CCL8",
      "CD14",
      "CD2",
      "CD40",
      "CD40LG",
      "CDKN1A",
      "CHUK",
      "CSNK2A1",
      "CSNK2B",
      "Cytokine Response",
      "EDA2R",
      "FAS",
      "FYN",
      "IFI44",
      "IGFBP3",
      "IKBKB",
      "IKBKG",
      "IL17D",
      "IL32",
      "IRAK2",
      "LYN",
      "NF-\u03baB Pathway",
      "RIPK1",
      "TLR4",
      "TNFRSF1A",
      "TNFRSF9",
      "Tau Aggregation",
      "VCAM1",
      "ZAP70"
    ],
    "Autophagy & Lysosome": [
      "PIK3C2B",
      "PIK3CA",
      "PIK3CD",
      "PIK3CG",
      "TFEB"
    ],
    "Mitochondrial Morphology": [
      "DGKQ",
      "ING1",
      "MFN1",
      "MFN2",
      "Mitochondrial Morphology",
      "PIK3C3",
      "PPIP5K1",
      "PPM1N",
      "TRIM24"
    ],
    "MAPK & Axon Guidance": [
      "Axon Guidance",
      "EFNA4",
      "EPHB2",
      "EPHB3",
      "ILK",
      "LIMK1",
      "LIMK2",
      "MAP3K7",
      "MAP4K1",
      "MAP4K4",
      "MAPK Signaling",
      "MAPK4",
      "MAPKAPK2",
      "MAPKAPK5",
      "MET",
      "NRP1",
      "PAK4",
      "PLXNA1",
      "PLXNA3",
      "PLXNA4",
      "PLXNB2",
      "PLXNB3",
      "PLXNC1",
      "ROCK1",
      "RYK",
      "SRC",
      "SSH2",
      "SSH3"
    ],
    "Gene Fingerprinting": [
      "ACVR1",
      "ACVR1C",
      "ACVR2A",
      "MARK2",
      "PIK3AP1",
      "PPP1R3A",
      "PPP1R3B",
      "PPP2R1B",
      "PPP2R5C",
      "PRKCG",
      "PXK",
      "SLK",
      "SNRK",
      "STRADA"
    ],
    "Fragile X & Lysosome": [
      "FMR1",
      "FXR2"
    ],
    "Other": [
      "NLRP3"
    ],
    "Neddylation & Aging": [
      "MLN4924",
      "NAE1",
      "Neddylation Pathway",
      "SENP8",
      "UBA3"
    ],
    "AD Neurodegeneration": [
      "APPswe/swe",
      "A\u03b242",
      "Neurodegeneration",
      "PSENM146V/M146V",
      "Tau",
      "pTau"
    ],
    "PD Neurodegeneration": [
      "LRRK2G2019S/G2019S"
    ],
    "Additional Hit Genes": [
      "CEP170B",
      "DNAJB11",
      "FAM76B",
      "NUDT6",
      "PPP1CB",
      "PPPIAL4E",
      "VPS36"
    ],
    "Aging Pathways": [
      "Cellular Aging",
      "Proteostasis"
    ],
    "Lysosome homeostasis": [
      "FIG4",
      "INPP5E",
      "LAMP1",
      "Lysosome homeostasis",
      "MTMR14"
    ],
    "Mitochondrial": [
      "DGKQ",
      "ING1",
      "MFN1",
      "MFN2",
      "Mitochondrial function",
      "PPIP5K1",
      "PPM1N",
      "Respiratory function",
      "TRIM24"
    ],
    "Inflammatory": [
      "CARD11",
      "CCL2",
      "CCL27",
      "CCL8",
      "CD14",
      "CD2",
      "CD40",
      "CD40LG",
      "CDKN1A",
      "CHUK",
      "CSNK2A1",
      "CSNK2B",
      "EDA2R",
      "FAS",
      "IFI44",
      "IGFBP3",
      "IKBKB",
      "IKBKG",
      "IL17D",
      "IL32",
      "IRAK2",
      "LYN",
      "RIPK1",
      "TLR4",
      "TNFRSF1A",
      "TNFRSF9",
      "VCAM1",
      "ZAP70"
    ],
    "Other signaling": [
      "ARAF",
      "CDK9",
      "FMR1",
      "FXR2",
      "HDAC4",
      "KDM7A",
      "LTK",
      "MAP3K7CL",
      "MAP3K9",
      "MTA3",
      "PFKFB4",
      "PGK1",
      "PHF11",
      "PKLR",
      "SETDB2",
      "SMARCA2"
    ],
    "Mitochondrial Morphology & Bioenergetics": [
      "Canonical glycolysis",
      "Complex I biogenesis",
      "DGKQ",
      "Elongated mitochondria",
      "Fragmented mitochondria",
      "IMPA1",
      "ING1",
      "MFN1",
      "MFN2",
      "Mitochondrial fatty acid beta-oxidation",
      "Mitochondrial function",
      "Mitochondrial membrane potential",
      "Mitochondrial morphology",
      "Mitochondrial translation",
      "Mitochondrial transport",
      "OCRL",
      "Oxidative phosphorylation",
      "Oxygen Consumption Rate (OCR)",
      "PPIP5K1",
      "PPM1N",
      "Respiratory electron transport chain",
      "Respiratory function",
      "Seahorse ATP Production Rate assay",
      "Seahorse Glycolytic Rate assay",
      "Seahorse Mito Stress Test",
      "TCA cycle",
      "TMRE labeling",
      "TRIM24"
    ]
  }
