# -*- coding: utf-8 -*-

"""Implementation of combinations for the :class:`pykeen.models.LiteralModel`."""

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

import torch
from class_resolver import HintOrType
from torch import nn

from ..utils import activation_resolver, combine_complex, split_complex

__all__ = [
    'Combination',
    'RealCombination',
    'ParameterizedRealCombination',
    'ComplexCombination',
    'ParameterizedComplexCombination',
    # Concrete classes
    'LinearDropout',
    'DistMultCombination',
    'ComplExLiteralCombination',
    'LinearV1',
    'TransEV1',
    'LinearV2',
    'TransEV2',
    'LinearV3',
    'TransEV3'
]


class Combination(nn.Module, ABC):
    """Base class for combinations."""

    def forward(self, x: torch.FloatTensor, literal: torch.FloatTensor) -> torch.FloatTensor:
        """Combine the representation and literal then score."""
        raise NotImplementedError


class RealCombination(Combination, ABC):
    """A mid-level base class for combinations of real-valued vectors."""

    def forward(self, x: torch.FloatTensor, literal: torch.FloatTensor) -> torch.FloatTensor:
        """Combine the entity representation and literal, then score."""
        return self.score(torch.cat([x, literal], dim=-1))

    @abstractmethod
    def score(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined entity representation and literals."""
        raise NotImplementedError


class ParameterizedRealCombination(RealCombination):
    """A real combination parametrized by a scoring module."""

    def __init__(self, module: nn.Module):
        """Initialize the parameterized real combination.

        :param module: The module used to score the combination of the entity representation and literals.
        """
        super().__init__()
        self.module = module

    def score(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined entity representation and literals with the parameterized module."""
        return self.module(x)


class ComplexCombination(Combination, ABC):
    """A mid-level base class for combinations of complex-valued vectors."""

    def forward(self, x: torch.FloatTensor, literal: torch.FloatTensor) -> torch.FloatTensor:
        """Split the complex vector, combine the representation parts and literal, score, then recombine."""
        x_re, x_im = split_complex(x)
        x_re = self.score_real(torch.cat([x_re, literal], dim=-1))
        x_im = self.score_imag(torch.cat([x_im, literal], dim=-1))
        return combine_complex(x_re=x_re, x_im=x_im)

    @abstractmethod
    def score_real(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined real part of the entity representation and literals."""
        raise NotImplementedError

    @abstractmethod
    def score_imag(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined imaginary part of the entity representation and literals."""
        raise NotImplementedError


class ParameterizedComplexCombination(ComplexCombination):
    """A complex combination parametrized by the real scoring module and imaginary soring module."""

    def __init__(self, real_module: nn.Module, imag_module: nn.Module):
        """Initialize the parameterized complex combination.

        :param real_module: The module used to score the combination of the real part of the entity representation
            and literals.
        :param imag_module: The module used to score the combination of the imaginary part of the entity
            representation and literals.
        """
        super().__init__()
        self.real_mod = real_module
        self.imag_mod = imag_module

    def score_real(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined real part of the entity representation and literals with the parameterized module."""
        return self.real_mod(x)

    def score_imag(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined imaginary part of the entity representation and literals with the parameterized module."""
        return self.imag_mod(x)


class LinearDropout(nn.Sequential):
    """A sequential module that has a linear layer, dropout later, and optional activation layer."""

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        input_dropout: float = 0.0,
        activation: HintOrType[nn.Module] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Instantiate the :class:`torch.nn.Sequential`.

        :param entity_embedding_dim: The dimension of the entity representations to which literals are concatenated
        :param literal_embedding_dim: The dimension of the literals that are concatenated
        :param input_dropout: The dropout probability of an element to be zeroed.
        :param activation: An optional, pre-instantiated activation module, like :class:`torch.nn.Tanh`.
        """
        linear = nn.Linear(entity_embedding_dim + literal_embedding_dim, entity_embedding_dim)
        dropout = nn.Dropout(input_dropout)
        if activation:
            activation_instance = activation_resolver.make(activation, activation_kwargs)
            super().__init__(linear, dropout, activation_instance)
        else:
            super().__init__(linear, dropout)




class LinearV1(nn.Sequential):
    """A sequential module that has a linear layer, dropout later, and optional activation layer."""

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        input_dropout: float = 0.0,
        activation: HintOrType[nn.Module] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Instantiate the :class:`torch.nn.Sequential`.

        :param entity_embedding_dim: The dimension of the entity representations to which literals are concatenated
        :param literal_embedding_dim: The dimension of the literals that are concatenated
        :param input_dropout: The dropout probability of an element to be zeroed.
        :param activation: An optional, pre-instantiated activation module, like :class:`torch.nn.Tanh`.
        """
        linear = nn.Linear(entity_embedding_dim + literal_embedding_dim, entity_embedding_dim)
        activation1=nn.Tanh()
        linear2 = nn.Linear(entity_embedding_dim, entity_embedding_dim)
        activation2=nn.Tanh()

        dropout = nn.Dropout(input_dropout)

        super().__init__(linear,activation1,linear2,activation2,dropout)





class LinearV2(nn.Module):
    """A sequential module that has a linear layer, dropout later, and optional activation layer."""

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        input_dropout: float = 0.0,
        activation: HintOrType[nn.Module] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        
        
        super().__init__()
        """Instantiate the :class:`torch.nn.Sequential`.

        :param entity_embedding_dim: The dimension of the entity representations to which literals are concatenated
        :param literal_embedding_dim: The dimension of the literals that are concatenated
        :param input_dropout: The dropout probability of an element to be zeroed.
        :param activation: An optional, pre-instantiated activation module, like :class:`torch.nn.Tanh`.
        """
        self.linear = nn.Linear(literal_embedding_dim, entity_embedding_dim)
        self.activation1=nn.Tanh()
        self.linear2 = nn.Linear(entity_embedding_dim, entity_embedding_dim)
        self.activation2=nn.Tanh()
        self.linear3 = nn.Linear(entity_embedding_dim, entity_embedding_dim)
        self.activation3=nn.Tanh()

        self.dropout = nn.Dropout(input_dropout)

#        super().__init__(linear,activation1,linear2,activation2,linear3,activation3,dropout)
    def forward(self,x,literal):
        literal=self.linear(literal)
        literal=self.activation1(literal)
        x=x+literal
        x=self.linear2(x)
        x=self.activation2(x)
        x=self.linear3(x)
        x=self.activation3(x)
        
        return x



class LinearV3(nn.Module):
    """A sequential module that has a linear layer, dropout later, and optional activation layer."""

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        input_dropout: float = 0.0,
        activation: HintOrType[nn.Module] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        
        
        super().__init__()
        """Instantiate the :class:`torch.nn.Sequential`.

        :param entity_embedding_dim: The dimension of the entity representations to which literals are concatenated
        :param literal_embedding_dim: The dimension of the literals that are concatenated
        :param input_dropout: The dropout probability of an element to be zeroed.
        :param activation: An optional, pre-instantiated activation module, like :class:`torch.nn.Tanh`.
        """
        self.linear = nn.Linear(entity_embedding_dim + literal_embedding_dim, entity_embedding_dim)
        self.activation1=nn.Tanh()
        self.linear2 = nn.Linear(literal_embedding_dim, entity_embedding_dim)
        self.activation2=nn.Tanh()
        self.linear3 = nn.Linear(entity_embedding_dim, entity_embedding_dim)
        self.activation3=nn.Tanh()
        self.gate_activation=nn.Sigmoid()


        self.dropout = nn.Dropout(input_dropout)

#        super().__init__(linear,activation1,linear2,activation2,linear3,activation3,dropout)
    def forward(self,x,literal):
        
        x1 = self.linear(torch.cat([x, literal], dim=-1))
        x1 =  self.activation1(x1)
        literal=self.linear2(literal)
        literal=self.activation2(literal)
        
        gate=self.linear3(x1+literal)
        gatevalue=self.gate_activation(gate)
        out=(1-gatevalue)*x+gatevalue*literal
        
        return out


class LinearV4(nn.Module):
    """A sequential module that has a linear layer, dropout later, and optional activation layer."""

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        input_dropout: float = 0.0,
        activation: HintOrType[nn.Module] = None,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        
        
        super().__init__()
        """Instantiate the :class:`torch.nn.Sequential`.

        :param entity_embedding_dim: The dimension of the entity representations to which literals are concatenated
        :param literal_embedding_dim: The dimension of the literals that are concatenated
        :param input_dropout: The dropout probability of an element to be zeroed.
        :param activation: An optional, pre-instantiated activation module, like :class:`torch.nn.Tanh`.
        """
        self.linear = nn.Linear(literal_embedding_dim, entity_embedding_dim)
        self.activation1=nn.Tanh()
        self.linear2 = nn.Linear(entity_embedding_dim, entity_embedding_dim)
        self.activation2=nn.Tanh()
        self.linear3 = nn.Linear(entity_embedding_dim, entity_embedding_dim)
        self.activation3=nn.Tanh()

        self.dropout = nn.Dropout(input_dropout)

#        super().__init__(linear,activation1,linear2,activation2,linear3,activation3,dropout)
    def forward(self,x,literal):
        literal=self.linear(literal)
        literal=self.activation1(literal)
        x=x+0.05*literal
        x=self.linear2(x)
        x=self.activation2(x)
        x=self.linear3(x)
        x=self.activation3(x)
        
        return x


class TransEV1(ParameterizedRealCombination):
    """The linear/dropout combination used in :class:`pykeen.models.DistMultLiteral`."""

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        input_dropout: float = 0.0,
    ) -> None:
        """Instantiate the :class:`ParameterizedRealCombination` with a :class:`LinearDropout`.

        :param entity_embedding_dim: The dimension of the entity representations to which literals are concatenated
        :param literal_embedding_dim: The dimension of the literals that are concatenated
        :param input_dropout: The dropout probability of an element to be zeroed.

        This class does not use an activation in the :class:`LinearDropout` as
        described by [kristiadi2018]_.
        """
        super().__init__(LinearV1(
            entity_embedding_dim=entity_embedding_dim,
            literal_embedding_dim=literal_embedding_dim,
            input_dropout=input_dropout,
        ))



class TransEV2(ParameterizedRealCombination):
    """The linear/dropout combination used in :class:`pykeen.models.DistMultLiteral`."""

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        input_dropout: float = 0.0,
    ) -> None:
        """Instantiate the :class:`ParameterizedRealCombination` with a :class:`LinearDropout`.

        :param entity_embedding_dim: The dimension of the entity representations to which literals are concatenated
        :param literal_embedding_dim: The dimension of the literals that are concatenated
        :param input_dropout: The dropout probability of an element to be zeroed.

        This class does not use an activation in the :class:`LinearDropout` as
        described by [kristiadi2018]_.
        """
        super().__init__(LinearV2(
            entity_embedding_dim=entity_embedding_dim,
            literal_embedding_dim=literal_embedding_dim,
            input_dropout=input_dropout,
        ))

    def forward(self, x: torch.FloatTensor, literal: torch.FloatTensor) -> torch.FloatTensor:
        """Combine the entity representation and literal, then score."""
        
        
        return self.score(x,literal)

    def score(self, x: torch.FloatTensor,literal:torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined entity representation and literals with the parameterized module."""
        return self.module(x,literal)




class TransEV3(ParameterizedRealCombination):
    """The linear/dropout combination used in :class:`pykeen.models.DistMultLiteral`."""

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        input_dropout: float = 0.0,
    ) -> None:
        """Instantiate the :class:`ParameterizedRealCombination` with a :class:`LinearDropout`.

        :param entity_embedding_dim: The dimension of the entity representations to which literals are concatenated
        :param literal_embedding_dim: The dimension of the literals that are concatenated
        :param input_dropout: The dropout probability of an element to be zeroed.

        This class does not use an activation in the :class:`LinearDropout` as
        described by [kristiadi2018]_.
        """
        super().__init__(LinearV3(
            entity_embedding_dim=entity_embedding_dim,
            literal_embedding_dim=literal_embedding_dim,
            input_dropout=input_dropout,
        ))

    def forward(self, x: torch.FloatTensor, literal: torch.FloatTensor) -> torch.FloatTensor:
        """Combine the entity representation and literal, then score."""
        
        
        return self.score(x,literal)

    def score(self, x: torch.FloatTensor,literal:torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined entity representation and literals with the parameterized module."""
        return self.module(x,literal)





class TransEV4(ParameterizedRealCombination):
    """The linear/dropout combination used in :class:`pykeen.models.DistMultLiteral`."""

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        input_dropout: float = 0.0,
    ) -> None:
        """Instantiate the :class:`ParameterizedRealCombination` with a :class:`LinearDropout`.

        :param entity_embedding_dim: The dimension of the entity representations to which literals are concatenated
        :param literal_embedding_dim: The dimension of the literals that are concatenated
        :param input_dropout: The dropout probability of an element to be zeroed.

        This class does not use an activation in the :class:`LinearDropout` as
        described by [kristiadi2018]_.
        """
        super().__init__(LinearV4(
            entity_embedding_dim=entity_embedding_dim,
            literal_embedding_dim=literal_embedding_dim,
            input_dropout=input_dropout,
        ))

    def forward(self, x: torch.FloatTensor, literal: torch.FloatTensor) -> torch.FloatTensor:
        """Combine the entity representation and literal, then score."""
        
        
        return self.score(x,literal)

    def score(self, x: torch.FloatTensor,literal:torch.FloatTensor) -> torch.FloatTensor:
        """Score the combined entity representation and literals with the parameterized module."""
        return self.module(x,literal)






class DistMultCombination(ParameterizedRealCombination):
    """The linear/dropout combination used in :class:`pykeen.models.DistMultLiteral`."""

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        input_dropout: float = 0.0,
    ) -> None:
        """Instantiate the :class:`ParameterizedRealCombination` with a :class:`LinearDropout`.

        :param entity_embedding_dim: The dimension of the entity representations to which literals are concatenated
        :param literal_embedding_dim: The dimension of the literals that are concatenated
        :param input_dropout: The dropout probability of an element to be zeroed.

        This class does not use an activation in the :class:`LinearDropout` as
        described by [kristiadi2018]_.
        """
        super().__init__(LinearDropout(
            entity_embedding_dim=entity_embedding_dim,
            literal_embedding_dim=literal_embedding_dim,
            input_dropout=input_dropout,
        ))






class ComplExLiteralCombination(ParameterizedComplexCombination):
    """The linear/dropout/tanh combination used in :class:`pykeen.models.ComplExLiteral`."""

    def __init__(
        self,
        entity_embedding_dim: int,
        literal_embedding_dim: int,
        input_dropout: float = 0.0,
        activation: HintOrType[nn.Module] = 'tanh',
    ) -> None:
        """Instantiate the :class:`ParameterizedComplexCombination` with a :class:`LinearDropout` for real and complex.

        :param entity_embedding_dim: The dimension of the entity representations to which literals are concatenated
        :param literal_embedding_dim: The dimension of the literals that are concatenated
        :param input_dropout: The dropout probability of an element to be zeroed.
        :param activation: The activation function, resolved by :data:`pykeen.utils.activation_resolver`.

        This class uses a :class:`torch.nn.Tanh` by default for the activation to the :class:`LinearDropout` as
        described by [kristiadi2018]_.
        """
        super().__init__(
            real_module=LinearDropout(
                entity_embedding_dim=entity_embedding_dim,
                literal_embedding_dim=literal_embedding_dim,
                input_dropout=input_dropout,
                
            ),
            imag_module=LinearDropout(
                entity_embedding_dim=entity_embedding_dim,
                literal_embedding_dim=literal_embedding_dim,
                input_dropout=input_dropout,
                
            ),
        )
