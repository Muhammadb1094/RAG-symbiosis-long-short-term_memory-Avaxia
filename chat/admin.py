from django.contrib import admin
from .models import *


class ConversationAdmin(admin.ModelAdmin):
    list_display = ('label', 'user', 'created_at' )

admin.site.register(Conversation, ConversationAdmin)
admin.site.register(Message)


