from enum import Enum

class Data(Enum):
    TEST_ADS = "./test/test_data_ads.csv"
    TEST_FEEDS = "./test/test_data_feeds.csv"
    TRAIN_ADS = "./train/train_data_ads.csv"
    TRAIN_FEEDS = "./train/train_data_feeds.csv"

# Note these columns are common among ads and feeds:
# u_feedLifeCycle
# u_refreshTimes
# u_newsCatInterestsST

GReaT_template = (
    "log_id is {log_id}, user_id is {user_id}, age is {age}, gender is {gender}, "
    "residence is {residence}, city is {city}, city_rank is {city_rank}, series_dev is {series_dev}, "
    "series_group is {series_group}, emui_dev is {emui_dev}, device_name is {device_name}, "
    "device_size is {device_size}, net_type is {net_type}, task_id is {task_id}, adv_id is {adv_id}, "
    "creat_type_cd is {creat_type_cd}, adv_prim_id is {adv_prim_id}, inter_type_cd is {inter_type_cd}, "
    "slot_id is {slot_id}, site_id is {site_id}, spread_app_id is {spread_app_id}, "
    "hispace_app_tags is {hispace_app_tags}, app_second_class is {app_second_class}, "
    "app_score is {app_score}, ad_click_list_v001 is {ad_click_list_v001}, ad_click_list_v002 is {ad_click_list_v002}, "
    "ad_click_list_v003 is {ad_click_list_v003}, ad_close_list_v001 is {ad_close_list_v001}, "
    "ad_close_list_v002 is {ad_close_list_v002}, ad_close_list_v003 is {ad_close_list_v003}, pt_d is {pt_d}, "
    "u_newsCatInterestsST is {u_newsCatInterestsST_x}, u_refreshTimes is {u_refreshTimes_x}, u_feedLifeCycle is {u_feedLifeCycle_x}, "
    "u_userId is {u_userId}, u_phonePrice is {u_phonePrice}, u_browserLifeCycle is {u_browserLifeCycle}, "
    "u_browserMode is {u_browserMode}, u_newsCatInterests is {u_newsCatInterests}, u_newsCatDislike is {u_newsCatDislike}, "
    "u_click_ca2_news is {u_click_ca2_news}, i_docId is {i_docId}, i_s_sourceId is {i_s_sourceId}, "
    "i_regionEntity is {i_regionEntity}, i_cat is {i_cat}, i_dislikeTimes is {i_dislikeTimes}, "
    "i_upTimes is {i_upTimes}, i_dtype is {i_dtype}, e_ch is {e_ch}, e_m is {e_m}, e_po is {e_po}, e_pl is {e_pl}, "
    "e_rn is {e_rn}, e_section is {e_section}, e_et is {e_et}, label is {label}, cillabel is {cillabel}, pro is {pro}"
)
