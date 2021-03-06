# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server.models.check_response_field import CheckResponseField  # noqa: F401,E501
from swagger_server import util


class CheckResponse(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, validation_result: str=None, id_num: CheckResponseField=None, birthdate: CheckResponseField=None, birthplace: CheckResponseField=None, name: CheckResponseField=None, mother_name: CheckResponseField=None, release_date: CheckResponseField=None):  # noqa: E501
        """CheckResponse - a model defined in Swagger

        :param validation_result: The validation_result of this CheckResponse.  # noqa: E501
        :type validation_result: str
        :param id_num: The id_num of this CheckResponse.  # noqa: E501
        :type id_num: CheckResponseField
        :param birthdate: The birthdate of this CheckResponse.  # noqa: E501
        :type birthdate: CheckResponseField
        :param birthplace: The birthplace of this CheckResponse.  # noqa: E501
        :type birthplace: CheckResponseField
        :param name: The name of this CheckResponse.  # noqa: E501
        :type name: CheckResponseField
        :param mother_name: The mother_name of this CheckResponse.  # noqa: E501
        :type mother_name: CheckResponseField
        :param release_date: The release_date of this CheckResponse.  # noqa: E501
        :type release_date: CheckResponseField
        """
        self.swagger_types = {
            'validation_result': str,
            'id_num': CheckResponseField,
            'birthdate': CheckResponseField,
            'birthplace': CheckResponseField,
            'name': CheckResponseField,
            'mother_name': CheckResponseField,
            'release_date': CheckResponseField
        }

        self.attribute_map = {
            'validation_result': 'validation_result',
            'id_num': 'id_num',
            'birthdate': 'birthdate',
            'birthplace': 'birthplace',
            'name': 'name',
            'mother_name': 'mother_name',
            'release_date': 'release_date'
        }

        self._validation_result = validation_result
        self._id_num = id_num
        self._birthdate = birthdate
        self._birthplace = birthplace
        self._name = name
        self._mother_name = mother_name
        self._release_date = release_date

    @classmethod
    def from_dict(cls, dikt) -> 'CheckResponse':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The CheckResponse of this CheckResponse.  # noqa: E501
        :rtype: CheckResponse
        """
        return util.deserialize_model(dikt, cls)

    @property
    def validation_result(self) -> str:
        """Gets the validation_result of this CheckResponse.


        :return: The validation_result of this CheckResponse.
        :rtype: str
        """
        return self._validation_result

    @validation_result.setter
    def validation_result(self, validation_result: str):
        """Sets the validation_result of this CheckResponse.


        :param validation_result: The validation_result of this CheckResponse.
        :type validation_result: str
        """
        allowed_values = ["accept", "validate", "reject"]  # noqa: E501
        if validation_result not in allowed_values:
            raise ValueError(
                "Invalid value for `validation_result` ({0}), must be one of {1}"
                .format(validation_result, allowed_values)
            )

        self._validation_result = validation_result

    @property
    def id_num(self) -> CheckResponseField:
        """Gets the id_num of this CheckResponse.


        :return: The id_num of this CheckResponse.
        :rtype: CheckResponseField
        """
        return self._id_num

    @id_num.setter
    def id_num(self, id_num: CheckResponseField):
        """Sets the id_num of this CheckResponse.


        :param id_num: The id_num of this CheckResponse.
        :type id_num: CheckResponseField
        """

        self._id_num = id_num

    @property
    def birthdate(self) -> CheckResponseField:
        """Gets the birthdate of this CheckResponse.


        :return: The birthdate of this CheckResponse.
        :rtype: CheckResponseField
        """
        return self._birthdate

    @birthdate.setter
    def birthdate(self, birthdate: CheckResponseField):
        """Sets the birthdate of this CheckResponse.


        :param birthdate: The birthdate of this CheckResponse.
        :type birthdate: CheckResponseField
        """

        self._birthdate = birthdate

    @property
    def birthplace(self) -> CheckResponseField:
        """Gets the birthplace of this CheckResponse.


        :return: The birthplace of this CheckResponse.
        :rtype: CheckResponseField
        """
        return self._birthplace

    @birthplace.setter
    def birthplace(self, birthplace: CheckResponseField):
        """Sets the birthplace of this CheckResponse.


        :param birthplace: The birthplace of this CheckResponse.
        :type birthplace: CheckResponseField
        """

        self._birthplace = birthplace

    @property
    def name(self) -> CheckResponseField:
        """Gets the name of this CheckResponse.


        :return: The name of this CheckResponse.
        :rtype: CheckResponseField
        """
        return self._name

    @name.setter
    def name(self, name: CheckResponseField):
        """Sets the name of this CheckResponse.


        :param name: The name of this CheckResponse.
        :type name: CheckResponseField
        """

        self._name = name

    @property
    def mother_name(self) -> CheckResponseField:
        """Gets the mother_name of this CheckResponse.


        :return: The mother_name of this CheckResponse.
        :rtype: CheckResponseField
        """
        return self._mother_name

    @mother_name.setter
    def mother_name(self, mother_name: CheckResponseField):
        """Sets the mother_name of this CheckResponse.


        :param mother_name: The mother_name of this CheckResponse.
        :type mother_name: CheckResponseField
        """

        self._mother_name = mother_name

    @property
    def release_date(self) -> CheckResponseField:
        """Gets the release_date of this CheckResponse.


        :return: The release_date of this CheckResponse.
        :rtype: CheckResponseField
        """
        return self._release_date

    @release_date.setter
    def release_date(self, release_date: CheckResponseField):
        """Sets the release_date of this CheckResponse.


        :param release_date: The release_date of this CheckResponse.
        :type release_date: CheckResponseField
        """

        self._release_date = release_date
