from django.urls import path
from . import views

app_name = 'story_generator'

urlpatterns = [
    path('', views.index, name='index'),
    path('generate/', views.generate_story, name='generate_story'),
    path('status/<uuid:story_id>/', views.story_status, name='story_status'),
    path('story/<uuid:story_id>/', views.story_detail, name='story_detail'),
]
