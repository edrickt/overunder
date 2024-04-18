from django.urls import path
from . import views

app_name = 'playerpoints'

urlpatterns = [
    path('', views.player_predictions, name='player_predictions'),
]
