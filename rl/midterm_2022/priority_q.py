import heapq 
# A simple implementation of Priority Queue
# using heapq.

class PriorityQueue(object):
    def __init__(self):
        self.queue = []
        self.elements = set()
  
    def __str__(self):
        # does not print in order
        return ' '.join([str(i) for i in self.queue])
  
    def isEmpty(self):
        return len(self.queue) == 0
    
    def contains(self, element):
        return element in self.elements
    
    def peek(self):
        assert len(self.queue) > 0
        return self.queue[0]
    
    def pop(self):
        assert len(self.queue) > 0
        popped_off = []
        next_priority, next_element = heapq.heappop(self.queue)
        self.elements.remove(next_element)
        return next_element
  
    def insert(self, element, priority):
        assert element not in self.elements
        heapq.heappush(self.queue, (priority, element))
        self.elements.add(element)
        
    def update(self, element, new_priority):
        assert element in self.elements
        self.delete(element)
        self.insert(element, new_priority)
        
    def delete(self, element):
        assert element in self.elements
        popped_off = []
        next_priority, next_element = heapq.heappop(self.queue)
        while next_element != element:
            popped_off.append((next_priority, next_element))
            next_priority, next_element = heapq.heappop(self.queue)
        self.elements.remove(element)
        for priority, elem in popped_off:
            heapq.heappush(self.queue, (priority, elem))