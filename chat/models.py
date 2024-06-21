from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_delete
from django.dispatch import receiver
from pymilvus import MilvusClient
from django.conf import settings


class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE,)
    label = models.TextField(max_length=200)
    last_message = models.TextField(default="No Last Message Found.")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE,)
    query = models.TextField(default="No Query Passed")
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"{self.conversation.label} - {self.query} -- {self.id}"

@receiver(post_delete, sender=Message)
def post_delete_message(sender, instance, **kwargs):
    client = MilvusClient(settings.MILVUS_DB_NAME)
    client.delete(collection_name="message", ids=[int(instance.id)])