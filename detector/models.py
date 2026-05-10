from django.db import models


class Transaction(models.Model):
    """Store transaction records and their fraud analysis results."""
    
    # Transaction details
    amount = models.FloatField(help_text="Transaction amount")
    source = models.CharField(max_length=50, help_text="Web, Mobile, ATM, POS")
    
    # Analysis results
    risk = models.IntegerField(default=0, help_text="Risk score 0-100")
    result = models.CharField(max_length=50, help_text="Fraud/Safe Transaction")
    explanation = models.TextField(blank=True, help_text="AI explanation")
    
    # Sender info
    sender_name = models.CharField(max_length=100, blank=True, default='')
    sender_acc = models.CharField(max_length=4, blank=True, default='')
    
    # Receiver info
    receiver_name = models.CharField(max_length=100, blank=True, default='')
    receiver_rel = models.CharField(max_length=50, blank=True, default='')
    
    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Transaction'
        verbose_name_plural = 'Transactions'

    def __str__(self):
        return f"₹{self.amount} - {self.result} ({self.risk}%)"
