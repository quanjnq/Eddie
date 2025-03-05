from selection.index_selection_evaluation import IndexSelection  # pragma: no cover
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

index_selection = IndexSelection()  # pragma: no cover
index_selection.run()  # pragma: no cover
