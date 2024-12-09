#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for ivadomed.losses

from math import isclose
import pytest
import torch

from ivadomed.losses import GeneralizedDiceLoss, MultiClassDiceLoss, TverskyLoss, FocalTverskyLoss,\
    DiceLoss, AdapWingLoss, L2loss, LossCombination, FocalDiceLoss


@pytest.mark.parametrize('params', [
    (torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]]]),
     torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]]]),
     -1.0,
     MultiClassDiceLoss(None)),

    (torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     -1.0,
     MultiClassDiceLoss(None)),

    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     -1 / 3,
     MultiClassDiceLoss(None)),

    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     -1,
     MultiClassDiceLoss(None)),

    (torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]]),
     -(1 + 1 / 2) / 2,
     MultiClassDiceLoss(None)),

    (torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]]),
     -(1 / 3),
     MultiClassDiceLoss(classes_of_interest=[1]))
])
def test_multiclassdiceloss(params):
    """Test MultiClassDiceLoss class in ivadomed.losses.

    Args:
        params (tuple): containing input tensor, target tensor, expected value, loss function
    """
    input, target, expected_value, loss_fct = params
    loss = loss_fct.forward(input, target)
    assert isclose(loss.detach().cpu().numpy(), expected_value, rel_tol=1e-3)


@pytest.mark.parametrize('params', [
    (torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]]]),
     torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]]]),
     -1.,
     GeneralizedDiceLoss(epsilon=1e-5, include_background=True)),
    (torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
     -0.8,
     GeneralizedDiceLoss(epsilon=1e-5, include_background=False)),
    (torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
     -11 / 16,
     GeneralizedDiceLoss(epsilon=1e-5)),
    (torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
     torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
     -1.,
     GeneralizedDiceLoss(epsilon=1e-5)),
    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
     -3 / 8,
     GeneralizedDiceLoss(epsilon=1e-5)),
    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
     0.0,
     GeneralizedDiceLoss(epsilon=1e-5, include_background=False)),
    (torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]]),
     -18 / 23,
     GeneralizedDiceLoss(epsilon=1e-5)),
    (torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]], [[[1.0, 0.0], [0.0, 1.0]]]]),
     torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 1.0]]]]),
     -0.8,
     GeneralizedDiceLoss(epsilon=1e-5, include_background=False)),
    (torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]], [[[1.0, 0.0], [0.0, 1.0]]]]),
     torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]], [[[1.0, 0.0], [1.0, 1.0]]]]),
     -11 / 16,
     GeneralizedDiceLoss(epsilon=1e-5, include_background=True))
])
def test_generalizeddiceloss(params):
    """Test GeneralizedDiceLoss class in ivadomed.losses.

    Args:
        params (tuple): containing input tensor, target tensor, expected value, loss function
    """
    input, target, expected_value, loss_fct = params
    loss = loss_fct.forward(input, target)
    assert isclose(loss.detach().cpu().numpy(), expected_value, rel_tol=1e-2)


@pytest.mark.parametrize('params', [
    (torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     -1.0,
     DiceLoss()),

    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     -1 / 3,
     DiceLoss()),

    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     -1,
     DiceLoss()),
])
def test_diceloss(params):
    """Test DiceLoss class in ivadomed.losses.

    Args:
        params (tuple): containing input tensor, target tensor, expected value, loss function
    """
    input, target, expected_value, loss_fct = params
    loss = loss_fct.forward(input, target)
    assert isclose(loss.detach().cpu().numpy(), expected_value, rel_tol=1e-2)


@pytest.mark.parametrize('params', [
    (torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]]]),
     torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]]]),
     -1.0,
     TverskyLoss(alpha=0.7, beta=0.3)),

    (torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     -1.0,
     TverskyLoss(alpha=0.7, beta=0.3)),

    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     -0.625,
     TverskyLoss(alpha=0.7, beta=0.3)),

    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     -0.417,
     TverskyLoss(alpha=0.3, beta=0.7, smooth=1.)),

    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     -0.0071,
     TverskyLoss(alpha=0.3, beta=0.7, smooth=0.01)),

    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     -1.,
     TverskyLoss(alpha=0.3, beta=0.7)),

    (torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]]),
     - (2 / (1 + 1) + (1 / (1 + 0.7))) / 2,
     TverskyLoss(alpha=0.3, beta=0.7)),
])
def test_tverskyloss(params):
    """Test TverskyLoss class in ivadomed.losses.

    Args:
        params (tuple): containing input tensor, target tensor, expected value, loss function
    """
    input, target, expected_value, loss_fct = params
    loss = loss_fct.forward(input, target)
    assert isclose(loss.detach().cpu().numpy(), expected_value, rel_tol=1e-2)


@pytest.mark.parametrize('params', [
    (torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]]]),
     torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]]]),
     0.,
     FocalTverskyLoss(alpha=0.7, beta=0.3)),

    (torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     0.,
     FocalTverskyLoss(alpha=0.7, beta=0.3)),

    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     0.52,
     FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.5)),

    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     0.375,
     FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1)),

    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     0.,
     FocalTverskyLoss(alpha=0.7, beta=0.3)),

    (torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]]),
     (pow(1 - (2 / (1 + 1)), 1 / 1.33) + pow(1 - (1 / (1 + 0.3)), 1 / 1.33)) / 2,
     FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)),
])
def test_focaltverskyloss(params):
    """Test FocalTverskyLoss class in ivadomed.losses.

    Args:
        params (tuple): containing input tensor, target tensor, expected value, loss function
    """
    input, target, expected_value, loss_fct = params
    loss = loss_fct.forward(input, target)
    assert isclose(loss.detach().cpu().numpy(), expected_value, rel_tol=1e-2)


@pytest.mark.parametrize('params', [
    (torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]]]),
     torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]]]),
     0.,
     L2loss()),

    (torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     0.,
     L2loss()),

    (torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
     1.0,
     L2loss()),

    (torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]]),
     torch.tensor([[[[0.5, 1.0], [1.0, 1.0]]]]),
     0.375,
     L2loss()),
])
def test_L2loss(params):
    """test L2Loss class in ivadomed.losses.

    Args:
        params (tuple): containing input tensor, target tensor, expected value, loss function

    """

    input, target, expected_value, loss_fct = params
    loss = loss_fct.forward(input, target)
    assert isclose(loss.detach().cpu().numpy(), expected_value, rel_tol=1e-2)


@pytest.mark.parametrize('params', [
    (torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
     0.,
     AdapWingLoss()),

    (torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
     29.0147,
     AdapWingLoss()),

    (torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
     41.4496,
     AdapWingLoss(omega=20)),

    (torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]]),
     torch.tensor([[[[0.5, 1.0], [1.0, 1.0]]]]),
     8.2703,
     AdapWingLoss(epsilon=2)),
])
def test_adapwingloss(params):
    """Test AdapWingLoss class in ivadomed.losses.

    Args:
        params (tuple): containing input tensor, target tensor, expected value, loss function

    """

    input, target, expected_value, loss_fct = params
    loss = loss_fct.forward(input, target)
    assert isclose(loss.detach().cpu().numpy(), expected_value, rel_tol=1e-2)


@pytest.mark.parametrize('params', [
    (torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
     -1.0,
     LossCombination(["DiceLoss", "L2loss"], [None, None])),

    (torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
     1.0,
     LossCombination(["DiceLoss", "L2loss"], [None, None])),

    (torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]]),
     torch.tensor([[[[0.5, 1.0], [1.0, 1.0]]]]),
     -0.3173,
     LossCombination(["DiceLoss", "L2loss"], [None, None])),
])
def test_losscombination(params):
    """Test LossCombination class in ivadomed.losses.

    Args:
        params (tuple): containing input tensor, target tensor, expected value, loss function

    """
    input, target, expected_value, loss_fct = params
    loss = loss_fct.forward(input, target)
    assert isclose(loss.detach().cpu().numpy(), expected_value, rel_tol=1e-2)


@pytest.mark.parametrize('params', [
    (torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
     -16.1181,
     FocalDiceLoss()),

    (torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
     torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
     3.6897,
     FocalDiceLoss()),

    (torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]]),
     torch.tensor([[[[0.5, 1.0], [1.0, 1.0]]]]),
     -1.1619,
     FocalDiceLoss()),
])
def test_focaldiceloss(params):
    """Test FocalDiceLoss class in ivadomed.losses.

    Args:
        params (tuple): containing input tensor, target tensor, expected value, loss function

    """

    input, target, expected_value, loss_fct = params
    loss = loss_fct.forward(input, target)
    assert isclose(loss.detach().cpu().numpy(), expected_value, rel_tol=1e-2)
