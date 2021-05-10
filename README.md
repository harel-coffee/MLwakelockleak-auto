# MLwakelockleak
Requirements:

	-> Networkx

	-> Numpy

	-> Androgurad

	-> TQDM

	-> skLearn

	-> Matplot

	-> imblearn

You need to change this to make it compatible

--- a/androguard/core/analysis/analysis.py
+++ b/androguard/core/analysis/analysis.py
@@ -1403,7 +1403,7 @@ class Analysis:
             """
             Wrapper to add methods to a graph
             """
-            if method not in G.node:
+            if method not in G.nodes:
                 if isinstance(method, ExternalMethod):
                     is_external = True
                 else:
