# -*- coding: utf-8 -*-
# AoT (And-or Tree) implementation for RAVEN task generation

import copy
import numpy as np
try:
    from scipy.special import comb
except ImportError:
    from math import comb

from .attributes import Angle, Color, Number, Position, Size, Type, Uniformity
from .constraints import rule_constraint


class AoTNode(object):
    """Superclass of AoT. 
    """

    levels_next = {"Root": "Structure",
                   "Structure": "Component",
                   "Component": "Layout",
                   "Layout": "Entity"}

    def __init__(self, name, level, node_type, is_pg=False):
        self.name = name
        self.level = level
        self.node_type = node_type
        self.children = []
        self.is_pg = is_pg
    
    def insert(self, node):
        """Used for public.
        Arguments:
            node(AoTNode): a node to insert
        """
        assert isinstance(node, AoTNode)
        assert self.node_type != "leaf"
        assert node.level == self.levels_next[self.level]
        self.children.append(node)
    
    def _insert(self, node):
        """Used for private.
        Arguments:
            node(AoTNode): a node to insert
        """
        assert isinstance(node, AoTNode)
        assert self.node_type != "leaf"
        assert node.level == self.levels_next[self.level]
        self.children.append(node)
    
    def _resample(self, change_number):
        """Resample the layout. If the number of entities change, resample also the 
        position distribution; otherwise only resample each attribute for each entity.
        Arugments:
            change_number(bool): whether to the number has been reset
        """
        assert self.is_pg
        if self.node_type == "and":
            for child in self.children:
                child._resample(change_number)
        else:
            self.children[0]._resample(change_number)

    def __repr__(self):
        return self.level + "." + self.name
    
    def __str__(self):
        return self.level + "." + self.name


class Root(AoTNode):

    def __init__(self, name, is_pg=False):
        super(Root, self).__init__(name, level="Root", node_type="or", is_pg=is_pg)

    def sample(self):
        """The function returns a separate AoT that is correctly parsed.
        Note that a new node is needed so that modification does not alter settings
        in the original tree.
        Returns:
            new_node(Root): a newly instantiated node
        """
        if self.is_pg:
            raise ValueError("Could not sample on a PG")
        new_node = Root(self.name, True)
        selected = np.random.choice(self.children)
        new_node.insert(selected._sample())
        return new_node

    def resample(self, change_number=False):
        self._resample(change_number)
    
    def prune(self, rule_groups):
        """Prune the AoT such that all branches satisfy the constraints. 
        Arguments:
            rule_groups(list of list of Rule): each list of Rule applies to a component
        Returns:
            new_node(Root): a newly instantiated node with branches all satisfying the constraints;
                None if no branches satisfy all the constraints
        """
        new_node = Root(self.name)
        for structure in self.children:
            if len(structure.children) == len(rule_groups):
                new_child = structure._prune(rule_groups)
                if new_child is not None:
                    new_node.insert(new_child)
        # during real execution, this should never happens
        if len(new_node.children) == 0:
            new_node = None
        return new_node
    
    def prepare(self):
        """This function prepares the AoT for rendering.
        Returns:
            structure.name(str): used for rendering structure
            entities(list of Entity): used for rendering each entity
        """
        assert self.is_pg
        assert self.level == "Root"
        structure = self.children[0]
        components = []
        for child in structure.children:
            components.append(child)
        entities = []
        for component in components:
            for child in component.children[0].children:
                entities.append(child)
        return structure.name, entities
    
    def sample_new(self, component_idx, attr_name, min_level, max_level, root):
        """Sample a new configuration. This is used for generating answers.
        Arguments:
            component_idx(int): the component we will sample
            attr_name(str): name of the attribute to sample
            min_level(int): lower bound of value level for the attribute
            max_level(int): upper bound of value level for the attribute
            root(AoTNode): the answer AoT, used for storing previous value levels for each attribute
        """
        assert self.is_pg
        self.children[0]._sample_new(component_idx, attr_name, min_level, max_level, root.children[0])


class Structure(AoTNode):

    def __init__(self, name, is_pg=False):
        super(Structure, self).__init__(name, level="Structure", node_type="and", is_pg=is_pg)
    
    def _sample(self):
        if self.is_pg:
            raise ValueError("Could not sample on a PG")
        new_node = Structure(self.name, True)
        for child in self.children:
            new_node.insert(child._sample())
        return new_node
    
    def _prune(self, rule_groups):
        new_node = Structure(self.name)
        for i in range(len(self.children)):
            child = self.children[i]
            # if any of the components fails to satisfy the constraint
            # the structure could not be chosen
            new_child = child._prune(rule_groups[i])
            if new_child is None:
                return None
            new_node.insert(new_child)
        return new_node
    
    def _sample_new(self, component_idx, attr_name, min_level, max_level, structure):
        self.children[component_idx]._sample_new(attr_name, min_level, max_level, structure.children[component_idx])
        

class Component(AoTNode):

    def __init__(self, name, is_pg=False):
        super(Component, self).__init__(name, level="Component", node_type="or", is_pg=is_pg)

    def _sample(self):
        if self.is_pg:
            raise ValueError("Could not sample on a PG")
        new_node = Component(self.name, True)
        selected = np.random.choice(self.children)
        new_node.insert(selected._sample())
        return new_node

    def _prune(self, rule_group):
        new_node = Component(self.name)
        for child in self.children:
            new_child = child._update_constraint(rule_group)
            if new_child is not None:
                new_node.insert(new_child)
        if len(new_node.children) == 0:
            new_node = None
        return new_node
    
    def _sample_new(self, attr_name, min_level, max_level, component):
        self.children[0]._sample_new(attr_name, min_level, max_level, component.children[0])


class Layout(AoTNode):
    """Layout is the highest level of the hierarchy that has attributes (Number, Position and Uniformity).
    To copy a Layout, please use deepcopy such that newly instantiated and separated attributes are created.
    """

    def __init__(self, name, layout_constraint, entity_constraint, 
                             orig_layout_constraint=None, orig_entity_constraint=None, 
                             sample_new_num_count=None, is_pg=False):
        super(Layout, self).__init__(name, level="Layout", node_type="and", is_pg=is_pg)
        self.layout_constraint = layout_constraint
        self.entity_constraint = entity_constraint
        self.number = Number(min_level=layout_constraint["Number"][0], max_level=layout_constraint["Number"][1])
        self.position = Position(pos_type=layout_constraint["Position"][0], pos_list=layout_constraint["Position"][1])
        self.uniformity = Uniformity(min_level=layout_constraint["Uni"][0], max_level=layout_constraint["Uni"][1])
        self.number.sample()
        self.position.sample(self.number.get_value())
        self.uniformity.sample()
        # store initial layout_constraint and entity_constraint for answer generation
        if orig_layout_constraint is None:
            self.orig_layout_constraint = copy.deepcopy(self.layout_constraint)
        else:
            self.orig_layout_constraint = orig_layout_constraint
        if orig_entity_constraint is None:
            self.orig_entity_constraint = copy.deepcopy(self.entity_constraint)
        else:
            self.orig_entity_constraint = orig_entity_constraint
        if sample_new_num_count is None:
            self.sample_new_num_count = dict()
            most_num = len(self.position.values)
            for i in range(layout_constraint["Number"][0], layout_constraint["Number"][1] + 1):
                self.sample_new_num_count[i] = [int(comb(most_num, i + 1)), []]
        else:
            self.sample_new_num_count = sample_new_num_count

    def add_new(self, *bboxes):
        """Add new entities into this level.
        Arguments:
            *bboxes(tuple of bbox): bboxes of new entities
        """
        name = self.number.get_value()
        uni = self.uniformity.get_value()
        for i in range(len(bboxes)):
            name += i
            bbox = bboxes[i]
            new_entity = copy.deepcopy(self.children[0])
            new_entity.name = str(name)
            new_entity.bbox = bbox
            if not uni:
                new_entity.resample()
            self._insert(new_entity)
    
    def resample(self, change_number=False):
        self._resample(change_number)
            
    def _sample(self):
        """Though Layout is an "and" node, we do not enumerate all possible configurations, but rather
        we treat it as a sampling process such that different configurtions are sampled. After the
        sampling, the lower level Entities are instantiated.
        Returns:
            new_node(Layout): a separated node with independent attributes
        """
        pos = self.position.get_value()
        new_node = copy.deepcopy(self)
        new_node.is_pg = True
        if self.uniformity.get_value():
            node = Entity(name=str(0), bbox=pos[0], entity_constraint=self.entity_constraint)
            new_node._insert(node)
            for i in range(1, len(pos)):
                bbox = pos[i]
                node = copy.deepcopy(node)
                node.name = str(i)
                node.bbox = bbox
                new_node._insert(node)
        else:
            for i in range(len(pos)):
                bbox = pos[i]
                node = Entity(name=str(i), bbox=bbox, entity_constraint=self.entity_constraint)
                new_node._insert(node)
        return new_node
        
    def _resample(self, change_number):
        """Resample each attribute for every child.
        This function is called across rows.
        Arguments:
            change_number(bool): whether to resample a number
        """
        if change_number:
            self.number.sample()
        del self.children[:]
        self.position.sample(self.number.get_value())
        pos = self.position.get_value()
        if self.uniformity.get_value():
            node = Entity(name=str(0), bbox=pos[0], entity_constraint=self.entity_constraint)
            self._insert(node)
            for i in range(1, len(pos)):
                bbox = pos[i]
                node = copy.deepcopy(node)
                node.name = str(i)
                node.bbox = bbox
                self._insert(node)
        else:
            for i in range(len(pos)):
                bbox = pos[i]
                node = Entity(name=str(i), bbox=bbox, entity_constraint=self.entity_constraint)
                self._insert(node)

    def _update_constraint(self, rule_group):
        """Update the constraint of the layout. If one constraint is not satisfied, return None 
        such that this structure is disgarded.
        Arguments:
            rule_group(list of Rule): all rules to apply to this layout
        Returns:
            Layout(Layout): a new Layout node with independent attributes
        """        
        num_min = self.layout_constraint["Number"][0]
        num_max = self.layout_constraint["Number"][1]
        uni_min = self.layout_constraint["Uni"][0]
        uni_max = self.layout_constraint["Uni"][1]
        type_min = self.entity_constraint["Type"][0]
        type_max = self.entity_constraint["Type"][1]
        size_min = self.entity_constraint["Size"][0]
        size_max = self.entity_constraint["Size"][1]
        color_min = self.entity_constraint["Color"][0]
        color_max = self.entity_constraint["Color"][1]
        new_constraints = rule_constraint(rule_group, num_min, num_max, 
                                                      uni_min, uni_max,
                                                      type_min, type_max,
                                                      size_min, size_max,
                                                      color_min, color_max)
        new_layout_constraint, new_entity_constraint = new_constraints
        new_num_min = new_layout_constraint["Number"][0]
        new_num_max = new_layout_constraint["Number"][1]
        if new_num_min > new_num_max:
            return None
        new_uni_min = new_layout_constraint["Uni"][0]
        new_uni_max = new_layout_constraint["Uni"][1]
        if new_uni_min > new_uni_max:
            return None
        new_type_min = new_entity_constraint["Type"][0]
        new_type_max = new_entity_constraint["Type"][1]
        if new_type_min > new_type_max:
            return None
        new_size_min = new_entity_constraint["Size"][0]
        new_size_max = new_entity_constraint["Size"][1]
        if new_size_min > new_size_max:
            return None
        new_color_min = new_entity_constraint["Color"][0]
        new_color_max = new_entity_constraint["Color"][1]                                    
        if new_color_min > new_color_max:
            return None

        new_layout_constraint = copy.deepcopy(self.layout_constraint)
        new_layout_constraint["Number"][:] = [new_num_min, new_num_max]
        new_layout_constraint["Uni"][:] = [new_uni_min, new_uni_max]
        
        new_entity_constraint = copy.deepcopy(self.entity_constraint)
        new_entity_constraint["Type"][:] = [new_type_min, new_type_max]
        new_entity_constraint["Size"][:] = [new_size_min, new_size_max]
        new_entity_constraint["Color"][:] = [new_color_min, new_color_max]
        return Layout(self.name, new_layout_constraint, new_entity_constraint,
                                 self.orig_layout_constraint, self.orig_entity_constraint,
                                 self.sample_new_num_count)
    
    def reset_constraint(self, attr):
        attr_name = attr.lower()
        instance = getattr(self, attr_name)
        instance.min_level = self.layout_constraint[attr][0]
        instance.max_level = self.layout_constraint[attr][1]
    
    def _sample_new(self, attr_name, min_level, max_level, layout):
        if attr_name == "Number":
            while True:
                value_level = self.number.sample_new(min_level, max_level)
                if layout.sample_new_num_count[value_level][0] == 0:
                    continue
                new_num = self.number.get_value(value_level)
                new_value_idx = self.position.sample_new(new_num)
                set_new_value_idx = set(new_value_idx)
                if set_new_value_idx not in layout.sample_new_num_count[value_level][1]:
                    layout.sample_new_num_count[value_level][0] -= 1
                    layout.sample_new_num_count[value_level][1].append(set_new_value_idx)
                    break
            self.number.set_value_level(value_level)
            self.position.set_value_idx(new_value_idx)
            pos = self.position.get_value()
            del self.children[:]
            for i in range(len(pos)):
                bbox = pos[i]
                node = Entity(name=str(i), bbox=bbox, entity_constraint=self.entity_constraint)
                self._insert(node)
        elif attr_name == "Position":
            new_value_idx = self.position.sample_new(self.number.get_value())
            layout.position.previous_values.append(new_value_idx)
            self.position.set_value_idx(new_value_idx)
            pos = self.position.get_value()
            for i in range(len(pos)):
                bbox = pos[i]
                self.children[i].bbox = bbox
        elif attr_name == "Type":
            for index in range(len(self.children)):
                new_value_level = self.children[index].type.sample_new(min_level, max_level)
                self.children[index].type.set_value_level(new_value_level)
                layout.children[index].type.previous_values.append(new_value_level)
        elif attr_name == "Size":
            for index in range(len(self.children)):
                new_value_level = self.children[index].size.sample_new(min_level, max_level)
                self.children[index].size.set_value_level(new_value_level)
                layout.children[index].size.previous_values.append(new_value_level)
        elif attr_name == "Color":
            for index in range(len(self.children)):
                new_value_level = self.children[index].color.sample_new(min_level, max_level)
                self.children[index].color.set_value_level(new_value_level)
                layout.children[index].color.previous_values.append(new_value_level)
        else:
            raise ValueError("Unsupported operation")


class Entity(AoTNode):

    def __init__(self, name, bbox, entity_constraint):
        super(Entity, self).__init__(name, level="Entity", node_type="leaf", is_pg=True)
        # Attributes
        # Sample each attribute such that the value lies in the admissible range
        # Otherwise, random sample
        self.entity_constraint = entity_constraint
        self.bbox = bbox
        self.type = Type(min_level=entity_constraint["Type"][0], max_level=entity_constraint["Type"][1])
        self.type.sample()
        self.size = Size(min_level=entity_constraint["Size"][0], max_level=entity_constraint["Size"][1])
        self.size.sample()
        self.color = Color(min_level=entity_constraint["Color"][0], max_level=entity_constraint["Color"][1])
        self.color.sample()
        self.angle = Angle(min_level=entity_constraint["Angle"][0], max_level=entity_constraint["Angle"][1])
        self.angle.sample()
    
    def reset_constraint(self, attr, min_level, max_level):
        attr_name = attr.lower()
        self.entity_constraint[attr][:] = [min_level, max_level]
        instance = getattr(self, attr_name)
        instance.min_level = min_level
        instance.max_level = max_level

    def resample(self):
        self.type.sample()
        self.size.sample()
        self.color.sample()
        self.angle.sample()