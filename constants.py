from enum import Enum

class Data(Enum):
    TEST_ADS = "./test/test_data_ads.csv"
    TEST_FEEDS = "./test/test_data_feeds.csv"
    TRAIN_ADS = "./train/train_data_ads.csv"
    TRAIN_FEEDS = "./train/train_data_feeds.csv"

day_splits = { 
    'morning': [(4, 12)], 
    'afternoon': [(12, 20)],
    'night': [(20, 24), (0, 4)],
}

list_type_columns = {
    "ad_click_list_v001": [],
    "ad_click_list_v002": [],
    "ad_click_list_v003": [],
    "ad_close_list_v001": [],
    "ad_close_list_v002": [],
    "ad_close_list_v003": [],
    "hispace_app_tags": [],
    "u_newsCatInterests": [],
    "u_newsCatDislike": [],
    "u_newsCatInterestsST": [],
    "u_click_ca2_news": [],
    "i_entities": [],
    "u_newsCatInterestsST":[]
}


# Note these columns are common among ads and feeds:
# u_feedLifeCycle
# u_refreshTimes
# u_newsCatInterestsST

GReaT_template_naive = (
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

GReaT_template_time_user_grouping = {
    "basic": "on {day} at time {time_category}, ",

    "target_template": "user_id is {user_id}, age is {age}, gender is {gender}, "
    "residence is {residence}, city is {city}, city_rank is {city_rank}, series_dev is {series_dev}, "
    "series_group is {series_group}, emui_dev is {emui_dev}, device_name is {device_name}, "
    "device_size is {device_size}, net_type is {net_type}, task_id is {task_id}, adv_id is {adv_id}, "
    "creat_type_cd is {creat_type_cd}, adv_prim_id is {adv_prim_id}, inter_type_cd is {inter_type_cd}, "
    "slot_id is {slot_id}, site_id is {site_id}, spread_app_id is {spread_app_id}, "
    "hispace_app_tags is {hispace_app_tags}, app_second_class is {app_second_class}, "
    "app_score is {app_score}, ad_click_list_v001 is {ad_click_list_v001}, ad_click_list_v002 is {ad_click_list_v002}, "
    "ad_click_list_v003 is {ad_click_list_v003}, ad_close_list_v001 is {ad_close_list_v001}, "
    "ad_close_list_v002 is {ad_close_list_v002}, ad_close_list_v003 is {ad_close_list_v003}, pt_d is {pt_d}, ",
    
    "source_template": 
    "u_userId is {u_userId}, u_phonePrice is {u_phonePrice}, u_browserLifeCycle is {u_browserLifeCycle}, "
    "u_newsCatInterestsST is {u_newsCatInterestsST}, u_refreshTimes is {u_refreshTimes}, u_feedLifeCycle is {u_feedLifeCycle}, "
    "u_browserMode is {u_browserMode}, u_newsCatInterests is {u_newsCatInterests}, u_newsCatDislike is {u_newsCatDislike}, "
    "u_click_ca2_news is {u_click_ca2_news}, i_docId is {i_docId}, i_s_sourceId is {i_s_sourceId}, "
    "i_regionEntity is {i_regionEntity}, i_cat is {i_cat}, i_dislikeTimes is {i_dislikeTimes}, "
    "i_upTimes is {i_upTimes}, i_dtype is {i_dtype}, e_ch is {e_ch}, e_m is {e_m}, e_po is {e_po}, e_pl is {e_pl}, "
    "e_rn is {e_rn}, e_section is {e_section}, e_et is {e_et}, label is {label}, cillabel is {cillabel}, pro is {pro}"
}

    
    

RANDOM_SEED = 42


"""
EXAMPLE

on 2022-06-10 at time night,user_id is 283637, age is 2, gender is 2, residence is 21, city is 313, city_rank is 4, series_dev is 31, series_group is 3, emui_dev is 21, device_name is 151, device_size is 2117, net_type is 7, task_id is 29047, adv_id is 14261, creat_type_cd is 8, adv_prim_id is 2066, inter_type_cd is 4, slot_id is 22, site_id is 1, spread_app_id is 114, hispace_app_tags is 43, app_second_class is 18, app_score is 10.0, ad_click_list_v001 is 25255, and 20967, and 12116, and 12989, and 22937, ad_click_list_v002 is 1518, and 1105, and 1774, and 1535, ad_click_list_v003 is 114, and 306, and 168, ad_close_list_v001 is 24107, ad_close_list_v002 is 1218, ad_close_list_v003 is 173, pt_d is 202206100328,u_userId is 283637, u_phonePrice is 14, u_browserLifeCycle is 17, u_newsCatInterestsST is 168, and 151, and 191, and 63, and 119, u_refreshTimes is 4, u_feedLifeCycle is 17, u_browserMode is 14, u_newsCatInterests is 10, and 191, and 65, and 206, and 203, u_newsCatDislike is 0, u_click_ca2_news is 79, and 168, and 191, and 63, and 192, i_docId is ca1d63a9345f410cf91b1a98fc6df51b0cbc2579, i_s_sourceId is b03e9739ced900576274991281d3a47ae3acd3cb, i_regionEntity is 0, i_cat is 79, i_dislikeTimes is 0, i_upTimes is 0, i_dtype is 12, e_ch is 19, e_m is 591, e_po is 2, e_pl is 866, e_rn is 2, e_section is 0, e_et is 202206102127, label is -1, cillabel is -1, pro is 0


"""