from pair_load import PairLoad

print("reading...")
data_container = PairLoad('./test/test_data_ads.csv', './test/test_data_feeds.csv')

merged = data_container.simple_merge()

template = (
    "User {user_id} with a device priced at {u_phonePrice} engaged with the browser for {u_browserLifeCycle} "
    "and used the {u_browserMode} mode. "
)

article_list = data_container.simple_generate_article_by_row(df=merged, template=template)
print(article_list)