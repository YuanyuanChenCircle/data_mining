from django.urls import path

from . import views
from polls.views import *


# name is used for a particular mapping(a line in urlpatterns)

# data
CSV_PATH = {
        "cweka": "polls\\data\\1_3cweka.csv",
        "chinese_stock": "polls\\data\\2_chinese_stock.csv",
        "license_plate": "polls\\data\\3_license_plate.csv",
        "hapiness": "polls\\data\\4_hapiness.csv",
        "iris": "polls\\data\\5_iris.csv",
        "adult": "polls\\data\\adult.csv",
        "car": "polls\\data\\car.csv",
        "filtered_retail": "polls\\data\\filtered_retail.csv",
        "AP_1":"polls\\data\\AP_1.csv",
        "mush":"polls\\data\\mush.csv"


    }




urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('index1/', views.index, name='index1'),
    path('login/', views.login_view, name='login'),
    path('register/',views.register_view, name="register_view"),

]

# urlpatterns += [path(r'^download/(?P.*)$', 'django.views.static.serve',"polls\\data\\5_iris.csv")]




urlpatterns += [
    path(r'labeled/', views.labeled, name='labeled')
]

urlpatterns += [
    path(r'unlabeled/', views.unlabeled, name='unlabeled')
]

urlpatterns += [
    path(r'iris_data_set/', views.iris_data_set, name='iris_data_set')
]


urlpatterns += [
    path(r'car_data_set/', views.car_data_set, name='car_data_set')
]
# pre_process table view
urlpatterns += [
    path(r'table/<slug:df_description>/', TableView.as_view(), name='table_url_parsing'),
]



# classification
urlpatterns += [
    path(r'CF/<slug:new_csv_store_name>/', ClassificationView.as_view(), name='classification'),
    path(r'CF_result/<slug:new_csv_store_name>/', CFViewResult.as_view(), name='CFresult')
]
# clustering
urlpatterns += [
    path(r'CR_result_data_view/', views.data_view, name='data_view')
]






urlpatterns += [
    path(r'CF_result_data_view/', views.cf_data_view, name='cf_data_view')
]

urlpatterns += [
    path(r'CR/<slug:new_csv_store_name>/', ClusteringView.as_view(), name='classification'),
    path(r'CR_result/<slug:new_csv_store_name>/', CR_result.as_view(), name='CFresult')
]




urlpatterns += [
    path(r'AR/<slug:new_csv_store_name>/', AssociationRuleView.as_view(), name='classification'),
    # path(r'CR_result/<slug:new_csv_store_name>/', CR_result.as_view(), name='CFresult')
]

urlpatterns += [
    path('delete_all_local_cache/', views.delete_local_cache, name='del_local_cache')
]


# upload csv file
urlpatterns += [
    path('upload/', views.upload_file, name='upload_url'),
    path('download/<slug:new_csv_store_name>/', views.download_file, name='download_url')
]

# urlpatterns += [
#     path('upload/success/',views.success_url, name='op_success'),
#     path('upload/fail_upload/', views.fail_upload, name = 'fail_uload')
# ]


# docs
urlpatterns += [
    path('docs/<slug:method>/<slug:doc_name>/',DocsView.as_view(),name='view_docs')
]

#############################################
urlpatterns += [
    path('Cf/<slug:cf_name>/',Cf_name.as_view(),name='Cf_name'),
    path('Cr/<slug:cf_name>/', Cr_name.as_view(), name='Cr_name'),
    path('Re/<slug:cf_name>/', Ae_name.as_view(), name='Ae_name')


]


# classification
urlpatterns += [
    path(r'cf/<slug:cf_m>/<slug:new_csv_store_name>/', ClassificationView1.as_view(), name='classification1'),
    # path(r'CF_result/<slug:new_csv_store_name>/', CFViewResult.as_view(), name='CFresult')
]

#聚类
urlpatterns += [
    path(r'cr/<slug:cr_m>/<slug:new_csv_store_name>/', ClusteringView1.as_view(), name='clustering1'),
    # path(r'CF_result/<slug:new_csv_store_name>/', CFViewResult.as_view(), name='CFresult')
]
#关联规则
urlpatterns += [
    path(r're/<slug:re_m>/<slug:new_csv_store_name>/', AssociationRuleView1.as_view(), name='clustering1'),
    # path(r'CF_result/<slug:new_csv_store_name>/', CFViewResult.as_view(), name='CFresult')
]
#登录
urlpatterns += [
    path('login/', views.login_view, name='login'),
    # path(r'CF_result/<slug:new_csv_store_name>/', CFViewResult.as_view(), name='CFresult')
]


