"""
Adaptive Traffic Light Control System - Optimized Version
Features:
• Graph-based road network (intersections as nodes, roads as edges)
• BFS connectivity check & Dijkstra routing
• DP optimal signal timing O(n×T²)
• Greedy (longest queue first) comparison
• Multi-intersection coordination
"""

import heapq
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import random


# ============================================================
# SECTION 1: ROAD NETWORK AS GRAPH
# ============================================================

class Direction(Enum):
    NORTH = "North"
    SOUTH = "South"
    EAST = "East"
    WEST = "West"


class LaneType(Enum):
    STRAIGHT = "straight"
    TURNING = "turning"


class SignalState(Enum):
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"


@dataclass
class Intersection:
    id: str
    name: str
    x: float
    y: float
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id


@dataclass
class Road:
    source: Intersection
    target: Intersection
    travel_time: float
    distance: float
    lanes: int
    speed_limit: float
    
    def __hash__(self):
        return hash((self.source.id, self.target.id))


class RoadNetworkGraph:
    def __init__(self):
        self.intersections: Dict[str, Intersection] = {}
        self.roads: Dict[str, List[Road]] = defaultdict(list)
        self.road_matrix: Dict[Tuple[str, str], Road] = {}
        
    def add_intersection(self, intersection: Intersection):
        self.intersections[intersection.id] = intersection
        
    def add_road(self, road: Road):
        self.roads[road.source.id].append(road)
        self.road_matrix[(road.source.id, road.target.id)] = road
        
    def add_two_way_road(self, road1: Road, road2: Road):
        self.add_road(road1)
        self.add_road(road2)
    
    def get_neighbors(self, intersection_id: str) -> List[Tuple[Intersection, float]]:
        neighbors = []
        for road in self.roads.get(intersection_id, []):
            neighbors.append((road.target, road.travel_time))
        return neighbors
    
    def get_intersection(self, intersection_id: str) -> Optional[Intersection]:
        return self.intersections.get(intersection_id)
    
    @property
    def num_vertices(self) -> int:
        return len(self.intersections)
    
    @property
    def num_edges(self) -> int:
        return sum(len(roads) for roads in self.roads.values())


# ============================================================
# SECTION 2: BFS FOR CONNECTIVITY CHECK
# ============================================================

class ConnectivityChecker:
    @staticmethod
    def is_connected(graph: RoadNetworkGraph, start_id: str = None) -> Tuple[bool, List[str]]:
        if not graph.intersections:
            return True, []
        
        if start_id is None:
            start_id = next(iter(graph.intersections.keys()))
        
        visited = set()
        queue = deque([start_id])
        visited.add(start_id)
        
        while queue:
            current = queue.popleft()
            for neighbor, _ in graph.get_neighbors(current):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append(neighbor.id)
        
        all_intersections = set(graph.intersections.keys())
        is_fully_connected = visited == all_intersections
        
        return is_fully_connected, list(visited)
    
    @staticmethod
    def find_connected_components(graph: RoadNetworkGraph) -> List[List[str]]:
        all_intersections = set(graph.intersections.keys())
        visited = set()
        components = []
        
        for intersection_id in all_intersections:
            if intersection_id not in visited:
                component = []
                queue = deque([intersection_id])
                visited.add(intersection_id)
                
                while queue:
                    current = queue.popleft()
                    component.append(current)
                    for neighbor, _ in graph.get_neighbors(current):
                        if neighbor.id not in visited:
                            visited.add(neighbor.id)
                            queue.append(neighbor.id)
                
                components.append(component)
        
        return components
    
    @staticmethod
    def shortest_path_bfs(graph: RoadNetworkGraph, start: str, target: str) -> Optional[List[str]]:
        if start not in graph.intersections or target not in graph.intersections:
            return None
        
        visited = {start: None}
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            if current == target:
                path = []
                while current is not None:
                    path.append(current)
                    current = visited[current]
                return list(reversed(path))
            
            for neighbor, _ in graph.get_neighbors(current):
                if neighbor.id not in visited:
                    visited[neighbor.id] = current
                    queue.append(neighbor.id)
        
        return None


# ============================================================
# SECTION 3: DIJKSTRA'S ALGORITHM FOR EMERGENCY ROUTING
# ============================================================

class EmergencyRouter:
    @staticmethod
    def dijkstra_shortest_path(
        graph: RoadNetworkGraph, 
        start_id: str, 
        target_id: str
    ) -> Tuple[Optional[List[str]], float]:
        if start_id not in graph.intersections or target_id not in graph.intersections:
            return None, float('inf')
        
        pq = [(0, start_id, [start_id])]
        visited = set()
        best_times = {start_id: 0}
        
        while pq:
            current_time, current_id, path = heapq.heappop(pq)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            if current_id == target_id:
                return path, current_time
            
            for neighbor, travel_time in graph.get_neighbors(current_id):
                if neighbor.id not in visited:
                    new_time = current_time + travel_time
                    
                    if neighbor.id not in best_times or new_time < best_times[neighbor.id]:
                        best_times[neighbor.id] = new_time
                        heapq.heappush(pq, (new_time, neighbor.id, path + [neighbor.id]))
        
        return None, float('inf')
    
    @staticmethod
    def dijkstra_all_distances(
        graph: RoadNetworkGraph, 
        start_id: str
    ) -> Dict[str, Tuple[float, Optional[str]]]:
        distances = {intersection_id: float('inf') for intersection_id in graph.intersections}
        previous = {intersection_id: None for intersection_id in graph.intersections}
        distances[start_id] = 0
        
        pq = [(0, start_id)]
        visited = set()
        
        while pq:
            current_dist, current_id = heapq.heappop(pq)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            for neighbor, travel_time in graph.get_neighbors(current_id):
                if neighbor.id not in visited:
                    new_dist = current_dist + travel_time
                    if new_dist < distances[neighbor.id]:
                        distances[neighbor.id] = new_dist
                        previous[neighbor.id] = current_id
                        heapq.heappush(pq, (new_dist, neighbor.id))
        
        return {intersection_id: (distances[intersection_id], previous[intersection_id]) 
                for intersection_id in graph.intersections}
    
    @staticmethod
    def find_emergency_route_with_preemption(
        graph: RoadNetworkGraph,
        emergency_location: str,
        destination: str,
        current_signals: Dict[str, SignalState]
    ) -> Dict:
        path, total_time = EmergencyRouter.dijkstra_shortest_path(
            graph, emergency_location, destination
        )
        
        if path is None:
            return {"error": "No path found"}
        
        return {
            "path": path,
            "total_travel_time": total_time,
            "num_intersections": len(path),
            "route_plan": [
                {"from": path[i], "to": path[i+1] if i+1 < len(path) else None, "preemption_required": True}
                for i in range(len(path))
            ]
        }


# ============================================================
# SECTION 3.5: DYNAMIC PROGRAMMING FOR OPTIMAL SIGNAL TIMING
# ============================================================

class DynamicProgrammingOptimizer:
    """
    DP for optimal signal timing based on historical queue data
    Recurrence: dp[i][g] = min(dp[i-1][prev_g] + Q[i] × g + λ|g - prev_g|)
    Time Complexity: O(n × T²)
    """
    
    @staticmethod
    def dp_optimal_timing(queue_lengths: List[int], max_green: int = 60, 
                          min_green: int = 10, step: int = 5) -> Tuple[List[int], float]:
        n = len(queue_lengths)
        T = (max_green - min_green) // step + 1
        green_options = [min_green + i * step for i in range(T)]
        
        dp = [[float('inf')] * T for _ in range(n)]
        choice = [[-1] * T for _ in range(n)]
        
        for g_idx, green_time in enumerate(green_options):
            dp[0][g_idx] = queue_lengths[0] * green_time
        
        for i in range(1, n):
            for g_idx, green_time in enumerate(green_options):
                for prev_idx in range(T):
                    transition_penalty = abs(green_options[prev_idx] - green_time) * 0.1
                    candidate = dp[i-1][prev_idx] + (queue_lengths[i] * green_time) + transition_penalty
                    
                    if candidate < dp[i][g_idx]:
                        dp[i][g_idx] = candidate
                        choice[i][g_idx] = prev_idx
        
        last_optimal_idx = min(range(T), key=lambda x: dp[n-1][x])
        min_total_wait = dp[n-1][last_optimal_idx]
        
        optimal_green_times = []
        current_idx = last_optimal_idx
        for i in range(n-1, -1, -1):
            optimal_green_times.insert(0, green_options[current_idx])
            if i > 0:
                current_idx = choice[i][current_idx]
        
        return optimal_green_times, min_total_wait


# ============================================================
# SECTION 3.6: GREEDY SHORTEST-JOB-FIRST
# ============================================================

class GreedySJFController:
    """Greedy approach - serves lane with longest queue first"""
    
    def __init__(self):
        self.name = "Greedy (Longest Queue First)"
    
    def select_next_direction(self, queue_lengths: Dict[Direction, int]) -> Direction:
        if not queue_lengths:
            return Direction.NORTH
        return max(queue_lengths, key=lambda d: queue_lengths.get(d, 0))
    
    def calculate_green_duration(self, queue_length: int, max_green: int = 60) -> int:
        return min(max_green, max(10, 10 + queue_length * 2))


# ============================================================
# SECTION 3.7: MULTI-INTERSECTION COORDINATION
# ============================================================

class MultiIntersectionCoordinator:
    """Coordinate multiple intersections using graph algorithms"""
    
    def __init__(self, graph: RoadNetworkGraph):
        self.graph = graph
        self.intersection_controllers: Dict[str, 'TrafficLightController'] = {}
        self.green_wave_pattern: Dict[str, int] = {}
    
    def register_intersection(self, intersection_id: str, controller: 'TrafficLightController'):
        self.intersection_controllers[intersection_id] = controller
    
    def calculate_green_wave_offsets(self, start_intersection: str, direction: Direction) -> Dict:
        offsets = {start_intersection: 0}
        distances = EmergencyRouter.dijkstra_all_distances(self.graph, start_intersection)
        
        for intersection_id, (distance, _) in distances.items():
            if intersection_id in self.intersection_controllers:
                offset = int(distance) % 60
                offsets[intersection_id] = offset
                self.green_wave_pattern[intersection_id] = offset
        
        return offsets
    
    def find_optimal_path_with_signals(self, start: str, end: str) -> Dict:
        if start not in self.graph.intersections or end not in self.graph.intersections:
            return {"error": "Invalid intersections"}
        
        pq = [(0, start, [start])]
        visited = set()
        best_times = {start: 0}
        
        while pq:
            current_time, current_id, path = heapq.heappop(pq)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            if current_id == end:
                return {"path": path, "total_travel_time": current_time, "num_intersections": len(path)}
            
            for neighbor, travel_time in self.graph.get_neighbors(current_id):
                if neighbor.id not in visited:
                    signal_penalty = self.green_wave_pattern.get(neighbor.id, 0) % 30
                    new_time = current_time + travel_time + signal_penalty
                    
                    if neighbor.id not in best_times or new_time < best_times[neighbor.id]:
                        best_times[neighbor.id] = new_time
                        heapq.heappush(pq, (new_time, neighbor.id, path + [neighbor.id]))
        
        return {"error": "No path found"}


# ============================================================
# SECTION 4: VEHICLE AND QUEUE MANAGEMENT
# ============================================================

class VehicleType(Enum):
    NORMAL = "normal"
    EMERGENCY = "emergency"


@dataclass
class Vehicle:
    vehicle_id: int
    lane_id: str
    arrival_timestamp: float
    vehicle_type: VehicleType = VehicleType.NORMAL
    
    @property
    def waiting_time(self) -> float:
        return time.time() - self.arrival_timestamp


class LaneQueue:
    def __init__(self, lane_id: str, max_length: int = 20):
        self.lane_id = lane_id
        self.queue: deque = deque()
        self.max_length = max_length
        self.total_vehicles_served = 0
        
    def enqueue(self, vehicle: Vehicle) -> bool:
        if len(self.queue) >= self.max_length:
            return False
        self.queue.append(vehicle)
        return True
    
    def dequeue(self) -> Optional[Vehicle]:
        if self.is_empty():
            return None
        self.total_vehicles_served += 1
        return self.queue.popleft()
    
    def peek(self) -> Optional[Vehicle]:
        return self.queue[0] if self.queue else None
    
    def is_empty(self) -> bool:
        return len(self.queue) == 0
    
    def size(self) -> int:
        return len(self.queue)
    
    def get_oldest_waiting_time(self) -> float:
        if self.is_empty():
            return 0
        return self.peek().waiting_time


class PriorityVehicleQueue:
    def __init__(self):
        self.priority_queue = []
        self.counter = 0
    
    def add_emergency_vehicle(self, vehicle: Vehicle) -> None:
        heapq.heappush(self.priority_queue, (0, self.counter, vehicle))
        self.counter += 1
    
    def add_normal_vehicle(self, vehicle: Vehicle) -> None:
        heapq.heappush(self.priority_queue, (1, self.counter, vehicle))
        self.counter += 1
    
    def get_next_vehicle(self) -> Optional[Vehicle]:
        if not self.priority_queue:
            return None
        return heapq.heappop(self.priority_queue)[2]
    
    def peek(self) -> Optional[Vehicle]:
        if not self.priority_queue:
            return None
        return self.priority_queue[0][2]
    
    def is_empty(self) -> bool:
        return len(self.priority_queue) == 0
    
    def size(self) -> int:
        return len(self.priority_queue)


# ============================================================
# SECTION 5: CIRCULAR BUFFER FOR TRAFFIC HISTORY
# ============================================================

@dataclass
class TrafficRecord:
    timestamp: float
    hour: int
    minute: int
    vehicle_counts: Dict[str, int]
    avg_waiting_times: Dict[str, float]


class CircularTrafficBuffer:
    def __init__(self, num_slots: int = 24, num_intersections: int = 4):
        self.num_slots = num_slots
        self.num_intersections = num_intersections
        self.buffer = [None] * num_slots
        self.head = 0
        self.length = 0
    
    def add_record(self, record: TrafficRecord):
        self.buffer[self.head] = record
        self.head = (self.head + 1) % self.num_slots
        if self.length < self.num_slots:
            self.length += 1
    
    def get_record(self, index: int) -> Optional[TrafficRecord]:
        if index >= self.length:
            return None
        oldest_index = (self.head - self.length + index) % self.num_slots
        return self.buffer[oldest_index]
    
    def get_queue_sequence(self, lane_id: str, hours: int = 24) -> List[int]:
        records = self.get_last_hours(hours)
        return [record.vehicle_counts.get(lane_id, 0) for record in records]
    
    def get_last_hours(self, hours: int) -> List[TrafficRecord]:
        hours = min(hours, self.length)
        records = []
        for i in range(self.length - hours, self.length):
            records.append(self.get_record(i))
        return records
    
    @property
    def memory_usage_bytes(self) -> int:
        return self.num_slots * 16 * self.num_intersections


# ============================================================
# SECTION 6: TRAFFIC LIGHT CONTROLLER
# ============================================================

class TrafficLightController:
    def __init__(self, intersection_id: str):
        self.intersection_id = intersection_id
        self.signals: Dict[Direction, SignalState] = {
            Direction.NORTH: SignalState.RED,
            Direction.SOUTH: SignalState.RED,
            Direction.EAST: SignalState.RED,
            Direction.WEST: SignalState.RED,
        }
        self.current_green: Optional[Direction] = None
        self.green_start_time: float = 0
        self.fixed_green_duration: float = 30
        self.fixed_yellow_duration: float = 3
        self.adaptive_green_duration: float = 30
        self.current_phase_state: str = 'green'
        self.emergency_preempted = False
        
        self.lanes: Dict[Direction, Dict[LaneType, LaneQueue]] = {}
        for direction in Direction:
            self.lanes[direction] = {
                LaneType.STRAIGHT: LaneQueue(f"{direction.value}_straight"),
                LaneType.TURNING: LaneQueue(f"{direction.value}_turning")
            }
        
        self.priority_queues: Dict[Direction, PriorityVehicleQueue] = {
            direction: PriorityVehicleQueue() for direction in Direction
        }
        
        # Optimization statistics
        self.dp_total_wait_time = 0
        self.greedy_total_wait_time = 0
        self.dp_vehicles_served = 0
        self.greedy_vehicles_served = 0
        self.greedy_controller = GreedySJFController()
    
    def add_vehicle(self, direction: Direction, lane_type: LaneType, is_emergency: bool = False) -> None:
        vehicle = Vehicle(
            vehicle_id=random.randint(10000, 99999),
            lane_id=f"{direction.value}_{lane_type.value}",
            arrival_timestamp=time.time(),
            vehicle_type=VehicleType.EMERGENCY if is_emergency else VehicleType.NORMAL
        )
        
        if is_emergency:
            self.priority_queues[direction].add_emergency_vehicle(vehicle)
            print(f"🚨 EMERGENCY VEHICLE added at {direction.value}")
            self.emergency_preempted = True
        else:
            self.lanes[direction][lane_type].enqueue(vehicle)
    
    def get_effective_queue_length(self, direction: Direction) -> int:
        return sum(q.size() for q in self.lanes[direction].values()) + self.priority_queues[direction].size()
    
    def update_fixed_time(self) -> Optional[Direction]:
        current_time = time.time()
        phases = list(Direction)
        
        if self.current_green is None:
            self.current_green = Direction.NORTH
            self.green_start_time = current_time
            self.current_phase_state = 'green'
            self._set_green(self.current_green)
            return self.current_green
        
        elapsed = current_time - self.green_start_time
        
        if self.current_phase_state == 'green':
            current_green_duration = self.adaptive_green_duration if self.emergency_preempted else self.fixed_green_duration
            if elapsed >= current_green_duration:
                self.current_phase_state = 'yellow'
                self.green_start_time = current_time
                self._set_yellow(self.current_green)
                return self.current_green
        
        elif self.current_phase_state == 'yellow':
            if elapsed >= self.fixed_yellow_duration:
                current_idx = phases.index(self.current_green)
                next_idx = (current_idx + 1) % len(phases)
                self.current_green = phases[next_idx]
                self.green_start_time = current_time
                self.current_phase_state = 'green'
                self._set_green(self.current_green)
                return self.current_green
        
        return self.current_green
    
    def update_greedy(self) -> Direction:
        queue_lengths = {d: self.get_effective_queue_length(d) for d in Direction}
        return self.greedy_controller.select_next_direction(queue_lengths)
    
    def update_dp_optimal(self, historical_data: List[Dict[str, int]]) -> int:
        if self.current_green:
            lane_id = f"{self.current_green.value}_straight"
            queue_sequence = [h.get(lane_id, 0) for h in historical_data]
            
            if len(queue_sequence) >= 2:
                optimal_times, _ = DynamicProgrammingOptimizer.dp_optimal_timing(
                    queue_sequence, max_green=60, min_green=10, step=5
                )
                if optimal_times:
                    return optimal_times[-1]
        return self.fixed_green_duration
    
    def _set_green(self, direction: Direction):
        for d in Direction:
            self.signals[d] = SignalState.RED
        self.signals[direction] = SignalState.GREEN
        print(f"🟢 GREEN light for {direction.value}")
    
    def _set_yellow(self, direction: Direction):
        self.signals[direction] = SignalState.YELLOW
        print(f"🟡 YELLOW light for {direction.value}")
    
    def serve_vehicles(self, direction: Direction, duration: float, method: str = "adaptive") -> int:
        vehicles_passed = 0
        max_vehicles = int(duration * 0.5)
        
        priority_queue = self.priority_queues[direction]
        lane_queues = self.lanes[direction]
        emergency_served = False
        
        while vehicles_passed < max_vehicles:
            vehicle = priority_queue.get_next_vehicle()
            if vehicle:
                if vehicle.vehicle_type == VehicleType.EMERGENCY:
                    print(f"  🚑 Emergency vehicle passed from {direction.value}")
                    emergency_served = True
                vehicles_passed += 1
                continue
            
            served = False
            for lane_type in [LaneType.STRAIGHT, LaneType.TURNING]:
                vehicle = lane_queues[lane_type].dequeue()
                if vehicle:
                    wait_time = vehicle.waiting_time
                    if method == "dp":
                        self.dp_total_wait_time += wait_time
                        self.dp_vehicles_served += 1
                    elif method == "greedy":
                        self.greedy_total_wait_time += wait_time
                        self.greedy_vehicles_served += 1
                    
                    print(f"  Vehicle passed from {direction.value} ({lane_type.value}) - waited {wait_time:.1f}s")
                    vehicles_passed += 1
                    served = True
                    break
            
            if not served:
                break
        
        if emergency_served:
            self.emergency_preempted = False
        
        return vehicles_passed
    
    def print_comparison_stats(self):
        print("\n" + "="*60)
        print("DP vs GREEDY COMPARISON RESULTS")
        print("="*60)
        
        dp_avg = self.dp_total_wait_time / max(1, self.dp_vehicles_served)
        greedy_avg = self.greedy_total_wait_time / max(1, self.greedy_vehicles_served)
        
        print(f"\n Dynamic Programming (DP) Approach:")
        print(f"   Vehicles served: {self.dp_vehicles_served}")
        print(f"   Avg waiting time: {dp_avg:.2f}s")
        
        print(f"\n Greedy (Longest Queue First) Approach:")
        print(f"   Vehicles served: {self.greedy_vehicles_served}")
        print(f"   Avg waiting time: {greedy_avg:.2f}s")
        
        if dp_avg > 0 and greedy_avg > 0:
            improvement = ((greedy_avg - dp_avg) / greedy_avg) * 100
            print(f"\nDP improvement: {improvement:.1f}% reduction in waiting time")
        
        print("\n DP Recurrence: dp[i][g] = min(dp[i-1][prev_g] + Q[i]×g + λ|g-prev_g|)")
        print("   Time Complexity: O(n × T²) where n=24, T=11 → ~2,904 ops")


# ============================================================
# SECTION 7: DEMO FUNCTIONS
# ============================================================

def create_sample_road_network() -> RoadNetworkGraph:
    graph = RoadNetworkGraph()
    
    intersections = [
        Intersection("A", "North Gate", 0, 100),
        Intersection("B", "Central", 0, 0),
        Intersection("C", "East Gate", 100, 0),
        Intersection("D", "South Gate", 0, -100),
        Intersection("E", "West Gate", -100, 0)
    ]
    
    for intersection in intersections:
        graph.add_intersection(intersection)
    
    roads = [
        (intersections[0], intersections[1], 45, 500),
        (intersections[1], intersections[2], 60, 700),
        (intersections[1], intersections[3], 55, 600),
        (intersections[1], intersections[4], 50, 550),
    ]
    
    for src, dst, time, dist in roads:
        graph.add_two_way_road(
            Road(src, dst, time, dist, 2, 40),
            Road(dst, src, time, dist, 2, 40)
        )
    
    return graph


def test_dp_optimization():
    print("\n" + "="*70)
    print("DYNAMIC PROGRAMMING FOR OPTIMAL SIGNAL TIMING")
    print("="*70)
    
    queue_data = [random.randint(2, 25) for _ in range(24)]
    print(f"\nHistorical Queue Data: {queue_data[:10]}...")
    
    optimal_times, min_wait = DynamicProgrammingOptimizer.dp_optimal_timing(queue_data)
    
    print(f"\n DP Optimal Green Durations (first 12 slots): {optimal_times[:12]}")
    print(f"   Minimum total waiting time: {min_wait:.1f}s")
    print(f"\n Complexity: O(24 × 11²) = 2,904 operations")


def test_greedy_vs_dp():
    print("\n" + "="*70)
    print("GREEDY vs DP COMPARISON")
    print("="*70)
    
    controller = TrafficLightController("Test Intersection")
    
    # Add test vehicles
    for _ in range(30):
        controller.add_vehicle(random.choice(list(Direction)), random.choice(list(LaneType)))
    
    # DP optimized
    historical = [{d.value.lower(): random.randint(5, 20) for d in Direction} for _ in range(24)]
    optimal_green = controller.update_dp_optimal(historical)
    print(f"\n🟢 DP Optimal green: {optimal_green}s")
    
    # Greedy
    greedy_dir = controller.update_greedy()
    print(f"🟢 Greedy selected: {greedy_dir.value}")
    
    controller.print_comparison_stats()


def test_multi_intersection():
    print("\n" + "="*70)
    print("MULTI-INTERSECTION COORDINATION")
    print("="*70)
    
    graph = create_sample_road_network()
    coordinator = MultiIntersectionCoordinator(graph)
    
    for intersection_id in graph.intersections.keys():
        coordinator.register_intersection(intersection_id, TrafficLightController(intersection_id))
    
    offsets = coordinator.calculate_green_wave_offsets("B", Direction.EAST)
    print(f"\n Green Wave Offsets: {offsets}")
    
    result = coordinator.find_optimal_path_with_signals("A", "D")
    if "path" in result:
        print(f"\n🛣️ Optimal Path: {' → '.join(result['path'])}")
        print(f"   Travel time: {result['total_travel_time']:.1f}s")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADAPTIVE TRAFFIC LIGHT CONTROL SYSTEM - OPTIMIZED")
    print("="*70)
    
    # Test DP
    test_dp_optimization()
    
    # Test Greedy vs DP
    test_greedy_vs_dp()
    
    # Test Multi-Intersection
    test_multi_intersection()
    
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    print("""
    DP Signal Timing: O(n × T²) = 2,904 operations
    reedy Baseline: O(D) = 4 operations  
    Multi-Intersection: O((V+E) log V) path planning
    Memory Optimized: Circular buffer O(24) slots
    Emergency Preemption: Priority queue O(log n)
    """)