from django import forms
from django_select2.forms import Select2Widget
from .models import Players
from overunder.models import TeamData

class PlayerPredictionForm(forms.Form):
    player_name = forms.ModelChoiceField(
        queryset=Players.objects.all(),
        widget=Select2Widget,
        empty_label=None,
        label='Player Name'
    )
    score_threshold = forms.FloatField(label='Score Over How Many Points')  # Changed from IntegerField to FloatField
    opponent_team = forms.ModelChoiceField(
        queryset=TeamData.objects.all(),
        widget=Select2Widget,
        empty_label="Any Team",
        required=False,
        label='Opponent Team'
    )
