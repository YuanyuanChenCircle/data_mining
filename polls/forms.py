from decimal import Decimal

from django import forms
from django.forms import RadioSelect
from django.utils.safestring import mark_safe
from django.contrib.auth import authenticate
from django.contrib.auth.models import User

"""
    if add new methods, need to update:
        urls.py
        add new template
        add new view
"""


class PreprocessForm(forms.Form):
    method_dict = {
        "CF": "Classification",
        "CR": "Clustering",
        "AR": "AssociationRules",
        "DL": "deeplearning"
    }
    Method_CHOICES = ((k, v) for k, v in method_dict.items())

    drop_missing = forms.BooleanField(required=False, help_text="Drop rows if there are missing slots.")
    char_to_digit = forms.BooleanField(required=False, help_text="Map non-numeric columns to numeric")
    method_selection = forms.ChoiceField(choices=Method_CHOICES)


class ClassificationForm(forms.Form):
    classfication_method_dict = {
        "LG": 'LogisticRegression',
        "KN": "KNeighborsClassifier",
        "SV": "SVC",
        # "GB": "GradientBoostingClassifier",
        "DT": "DecisionTreeClassifier",
        "RF": "RandomForestClassifier",
        "MP": "MLPClassifier",
        "NB": "GaussianNB",
    }
    dict_train_raito = {
        "0.1": "0.1",
        "0.2": "0.2",
        "0.3": "0.3",
        "0.4": "0.4",
        "0.5": "0.5",
        "0.6": "0.6",
        "0.7": "0.7",
        "0.8": "0.8",
        "0.9": "0.9"
    }
    Class_Train_Ratio = ((k,v) for k,v in dict_train_raito.items())
    Class_method_Choice = ((k, v) for k, v in classfication_method_dict.items())
    Classifier = forms.ChoiceField(choices=Class_method_Choice,
                                   widget=forms.Select(attrs={'onchange': 'ajax_class_change();'})
                                   )
    classification_parameters = forms.CharField(widget=forms.Textarea, required=False)
    label_name = forms.CharField()

    train_ratio = forms.ChoiceField(choices=Class_Train_Ratio)

    # target_column = forms.CharField(widget=forms.TextInput, required=True)

# class DelCacheForm(forms.Form):
#     delete_cache = forms.BooleanField(required=True)


class ClusteringForm(forms.Form):
    clustering_method_dict = {
        "KMS": 'KMeans',
        "MBKM": "MiniBatchKMeans",
        "AFP": "AffinityPropagation",
        # "GB": "GradientBoostingClassifier",
        "MSF": "MeanShift",
        "SPECC":"SpectralClustering",
        "AGC":"AgglomerativeClustering",
        "DBSCAN":"DBSCAN",
        "BRC":"Birch"
    }

    cluster_method_choice = ((k, v) for k, v in clustering_method_dict.items())
    Cluster_Algo = forms.ChoiceField(choices=cluster_method_choice,
                                   widget=forms.Select(attrs={'onchange': 'ajax_cluster_change();'})
                                   )
    Clustering_Parameters = forms.CharField(widget=forms.Textarea, required=False)


class ArForm(forms.Form):
    AR_method_dict = {
        "AR": 'Apriori',
        "FP": "FP",

    }

    AR_method_choice = ((k, v) for k, v in AR_method_dict.items())
    AR_Algo = forms.ChoiceField(choices=AR_method_choice,
                                   widget=forms.Select(attrs={'onchange': 'ajax_cluster_change();'})
                                   )
    AR_Parameters = forms.CharField(widget=forms.Textarea, required=False)





class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField(widget=forms.FileInput(attrs={'accept': ".csv"}))

########################

class PreprocessForm1(forms.Form):
    method_dict = {
        "1_3cweka": "1_3cweka",
        "5_iris": "5_iris",
        "adult": "adult",
        "car": "car"
    }
    Method_CHOICES = ((k, v) for k, v in method_dict.items())

    drop_missing = forms.BooleanField(required=False, help_text="是否删除数据中具有缺失值的行")
    char_to_digit = forms.BooleanField(required=False, help_text="是否对非数字列映射为数字")
    method_selection = forms.ChoiceField(choices=Method_CHOICES)


class ClassificationForm1(forms.Form):
    classfication_method_dict = {
        "LogisticRegression": 'LogisticRegression',
        "KNN": "KNeighborsClassifier",
        "SVC": "SVC",
        # "GB": "GradientBoostingClassifier",
        "DecisionTree": "DecisionTreeClassifier",
        "RandomForest": "RandomForestClassifier",
        "MLP": "MLPClassifier",
        "GaussianNB": "GaussianNB",
        # "GradientBoosting": "GradientBoostingClassifier",

    }
    dict_train_raito = {
        "0.1": "0.1",
        "0.2": "0.2",
        "0.3": "0.3",
        "0.4": "0.4",
        "0.5": "0.5",
        "0.6": "0.6",
        "0.7": "0.7",
        "0.8": "0.8",
        "0.9": "0.9"
    }
    Class_Train_Ratio = ((k,v) for k,v in dict_train_raito.items())
    Class_method_Choice = ((k, v) for k, v in classfication_method_dict.items())
    Classifier = forms.ChoiceField(choices=Class_method_Choice,
                                   widget=forms.Select(attrs={'onchange': 'ajax_class_change();'})
                                   )
    # Classifier = forms.CharField(choices=Class_method_Choice, widget=forms.Select(attrs={'onchange': 'ajax_class_change();'}), required=True
    #                                )
    classification_parameters = forms.CharField(widget=forms.Textarea, required=False)
    label_name = forms.CharField()

    train_ratio = forms.ChoiceField(choices=Class_Train_Ratio)

    # target_column = forms.CharField(widget=forms.TextInput, required=True)

class PreprocessForm2(forms.Form):
    method_dict = {
        "2_chinese_stock": "2_chinese_stock",
        "3_license_plate": "3_license_plate",
        "4_hapiness": "4_hapiness",
        "filtered_retail": "filtered_retail"
    }
    Method_CHOICES = ((k, v) for k, v in method_dict.items())

    drop_missing = forms.BooleanField(required=False, help_text="Drop rows if there are missing slots.")
    char_to_digit = forms.BooleanField(required=False, help_text="Map non-numeric columns to numeric")
    method_selection = forms.ChoiceField(choices=Method_CHOICES)



class ClusteringForm1(forms.Form):
    clustering_method_dict = {
        "KMeans": 'KMeans',
        "MiniBatchKMeans": "MiniBatchKMeans",
        "AffinityPropagation": "AffinityPropagation",
        # "GB": "GradientBoostingClassifier",
        "MeanShift": "MeanShift",
        "SpectralClustering":"SpectralClustering",
        "AgglomerativeClustering":"AgglomerativeClustering",
        "DBSCAN":"DBSCAN",
        "Birch":"Birch"
    }

    cluster_method_choice = ((k, v) for k, v in clustering_method_dict.items())
    Cluster_Algo = forms.ChoiceField(choices=cluster_method_choice,
                                   widget=forms.Select(attrs={'onchange': 'ajax_cluster_change();'})
                                   )
    Clustering_Parameters = forms.CharField(widget=forms.Textarea, required=False)


class PreprocessForm3(forms.Form):
    method_dict = {
        "mush": "mush"
        # "3_license_plate": "3_license_plate",
        # "4_hapiness": "4_hapiness",
        # "filtered_retail": "filtered_retail"
    }
    Method_CHOICES = ((k, v) for k, v in method_dict.items())

    drop_missing = forms.BooleanField(required=False, help_text="Drop rows if there are missing slots.")
    char_to_digit = forms.BooleanField(required=False, help_text="Map non-numeric columns to numeric")
    method_selection = forms.ChoiceField(choices=Method_CHOICES)


class LoginForm(forms.Form):
    username = forms.CharField(max_length=100, label='Username')
    password = forms.CharField(max_length=100, label='Password', widget=forms.PasswordInput)

    def clean(self):
        username = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')
        if username and password:
            user = authenticate(username=username, password=password)
            if not user:
                raise forms.ValidationError("invalid entrance!")
        return super(LoginForm, self).clean()


class RegisterForm(forms.ModelForm):
    username = forms.CharField(max_length=100, label='username')
    password = forms.CharField(max_length=100, label='password', widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = [
            'username',
            'first_name',
            'last_name',
            'email',

        ]
# class DelCacheForm(forms.Form):
#     delete_cache = forms.BooleanField(required=True)