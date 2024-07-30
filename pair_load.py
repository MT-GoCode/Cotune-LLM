import pandas as pd
import functools

class PairLoad():

    def __init__(self, target, source):
        self.target = pd.read_csv(target)
        self.source = pd.read_csv(source)

    def simple_merge(self):
        return pd.merge(self.target, self.source, right_on='u_userId', left_on='user_id')

    def simple_generate_article_by_row(self, df, template):
        def gen_row_sentence(row):
            return template.format(**row)
        
        return df.apply(lambda row: gen_row_sentence(row.to_dict()), axis=1).tolist()
