# forms.py in your playerpoints app

from django import forms
from django_select2.forms import Select2Widget
from .models import Players
from overunder.models import TeamData

class PlayerPredictionForm(forms.Form):
    player_name = forms.ModelChoiceField(
        queryset=Players.objects.all(),
        label='Player Name',
        widget=Select2Widget,
        empty_label="Select a Player"
    )
    score_threshold = forms.FloatField(label='Score Over How Many Points')  
    opponent_team = forms.ModelChoiceField(
        queryset=TeamData.objects.all(),
        label="Opponent",
        widget=Select2Widget,
        empty_label="Select a Team",
        required=False,
    )
