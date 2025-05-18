"""
Pagination helper functions for the PostgreSQL storage manager.
"""
import logging
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Tuple, Iterator, Union

# Get logger
logger = logging.getLogger('a2bot')

class Paginator:
    """Helper class to handle database pagination"""
    
    def __init__(self, session: Session, model_class, page_size=50):
        """
        Initialize the paginator
        
        Args:
            session: SQLAlchemy session
            model_class: The model class to query
            page_size: Number of records per page
        """
        self.session = session
        self.model_class = model_class
        self.page_size = page_size
        self.total_count = self._get_total_count()
        self.total_pages = (self.total_count + page_size - 1) // page_size if self.total_count > 0 else 0
        
    def _get_total_count(self) -> int:
        """Get the total count of records"""
        return self.session.query(func.count(self.model_class.id)).scalar() or 0
        
    def get_page(self, page_number: int) -> List[Any]:
        """
        Get a specific page of records
        
        Args:
            page_number: The page number (1-based)
            
        Returns:
            List of records for the requested page
        """
        if page_number < 1 or (self.total_pages > 0 and page_number > self.total_pages):
            return []
            
        offset = (page_number - 1) * self.page_size
        return self.session.query(self.model_class).order_by(self.model_class.id).offset(offset).limit(self.page_size).all()
        
    def get_all_pages(self) -> Iterator[List[Any]]:
        """
        Get all pages as an iterator
        
        Returns:
            Iterator yielding each page of records
        """
        for page_num in range(1, self.total_pages + 1):
            yield self.get_page(page_num)
            
    def get_info(self) -> Dict[str, int]:
        """
        Get pagination information
        
        Returns:
            Dictionary with pagination info
        """
        return {
            "total_records": self.total_count,
            "total_pages": self.total_pages,
            "page_size": self.page_size
        }
        
class BatchProcessor:
    """Helper class to process large datasets in batches"""
    
    def __init__(self, session: Session, model_class, batch_size=100):
        """
        Initialize the batch processor
        
        Args:
            session: SQLAlchemy session
            model_class: The model class to query
            batch_size: Number of records per batch
        """
        self.session = session
        self.model_class = model_class
        self.batch_size = batch_size
        
    def process_all(self, processor_func, query_filter=None, order_by=None):
        """
        Process all records in batches
        
        Args:
            processor_func: Function to process each batch of records
            query_filter: Optional filter to apply to the query
            order_by: Optional ordering to apply to the query
            
        Returns:
            Number of records processed
        """
        query = self.session.query(self.model_class)
        
        if query_filter is not None:
            query = query.filter(query_filter)
            
        if order_by is not None:
            query = query.order_by(order_by)
            
        total_processed = 0
        offset = 0
        
        while True:
            batch = query.limit(self.batch_size).offset(offset).all()
            if not batch:
                break
                
            processor_func(batch)
            total_processed += len(batch)
            offset += self.batch_size
            
            # Log progress periodically
            if total_processed % (self.batch_size * 10) == 0:
                logger.info(f"Processed {total_processed} records...")
                
        return total_processed
